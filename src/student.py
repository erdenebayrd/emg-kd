# src/models.py
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
    No session embedding (see thesis H1 methodology)."""
    def __init__(self, d_model=256, n_enc=4, n_down=3, in_ch=8,
                 vocab=VOCAB, n_head=4, ff_mult=4, dropout=0.1):
        super().__init__()
        assert n_down == 3, "keep 8x downsample for teacher-frame alignment in v1"
        chans = [in_ch] + [d_model] * n_down
        self.conv = nn.Sequential(*[
            ResConvBlock(chans[i], chans[i + 1], stride=2) for i in range(n_down)])
        enc = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(enc, n_enc)
        self.head = nn.Linear(d_model, vocab)
        self.downsample = 2 ** n_down

    def forward(self, x_raw):                 # (B, T, 8)
        x = self.conv(x_raw.transpose(1, 2))  # (B, d, T/8)
        x = x.transpose(1, 2)                 # (B, T/8, d)
        x = self.encoder(x)
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