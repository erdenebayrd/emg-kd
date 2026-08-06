# src/student.py
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 38          # 37 chars + CTC blank — must match teacher
BLANK = 37
DOWNSAMPLE = 8      # three stride-2 blocks, matches teacher frame rate


class ResConvBlock(nn.Module):
    """Stride-2 residual conv. Same time arithmetic as the teacher's ResBlock
    (kernel 3, stride 2, pad 1) so student frames align with teacher frames."""
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.down  = (nn.Conv1d(c_in, c_out, 1, stride=stride)
                      if (stride != 1 or c_in != c_out) else nn.Identity())

    def forward(self, x):                     # (B, C_in, T)
        r = self.down(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)                  # (B, C_out, T/stride)


class Student(nn.Module):
    """Compact CTC recogniser. Input raw EMG (B, T, 8) -> logits (B, T/8, 38).
    Fixed per-channel input normalization (padding-safe). No session embedding.

    Pass emg_mean / emg_std (each shape (8,)) computed once over real data.
    If not given, defaults to mean 0 / std 1 (no-op) — but you should pass them.
    """
    def __init__(self, d_model=256, n_enc=4, n_down=3, in_ch=8,
                 vocab=VOCAB, n_head=4, ff_mult=4, dropout=0.1,
                 emg_mean=None, emg_std=None):
        super().__init__()
        assert n_down == 3, "keep 8x downsample for teacher-frame alignment in v1"

        # fixed input normalization constants (buffers: saved with the model,
        # moved to device automatically, never updated by training)
        if emg_mean is None:
            emg_mean = torch.zeros(in_ch)
        if emg_std is None:
            emg_std = torch.ones(in_ch)
        self.register_buffer("emg_mean", torch.as_tensor(emg_mean).float().view(1, in_ch, 1))
        self.register_buffer("emg_std",  torch.as_tensor(emg_std).float().view(1, in_ch, 1))

        chans = [in_ch] + [d_model] * n_down
        self.conv = nn.Sequential(*[
            ResConvBlock(chans[i], chans[i + 1], stride=2) for i in range(n_down)])
        enc = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.head = nn.Linear(d_model, vocab)
        self.downsample = 2 ** n_down
        self._d_model = d_model

    def forward(self, x_raw, in_lens=None):   # x_raw: (B, T, 8)
        x = x_raw.transpose(1, 2)             # (B, 8, T)
        x = (x - self.emg_mean) / self.emg_std   # fixed normalization, padding-safe
        x = self.conv(x)                      # (B, d, T/8)
        x = x.transpose(1, 2)                 # (B, T/8, d)

        # padding mask for the transformer: True = ignore this frame
        mask = None
        if in_lens is not None:
            Tp = x.shape[1]
            ar = torch.arange(Tp, device=x.device)[None, :]      # (1, T/8)
            mask = ar >= in_lens[:, None]                         # (B, T/8) True where padded

        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(x)                   # (B, T/8, 38)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for cfg in [dict(d_model=384, n_enc=6),
                dict(d_model=256, n_enc=4),
                dict(d_model=128, n_enc=4)]:
        m = Student(**cfg).eval()
        with torch.no_grad():
            y = m(torch.randn(2, 800, 8))
        print(f"{cfg}  {n_params(m)/1e6:5.2f}M  out={tuple(y.shape)}")