# src/kd_data.py
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class KDDataset(Dataset):
    """Serves (raw_emg, teacher_logits, label) per utterance.
    raw_emg comes from Gaddy's EMGDataset (already preprocessed).
    teacher_logits + labels come from the cached files, aligned by index."""
    def __init__(self, emg_dataset, cache_dir, labels_path):
        self.ds = emg_dataset
        self.cache_dir = cache_dir
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)
        assert len(self.labels) == len(self.ds), \
            f"labels {len(self.labels)} != dataset {len(self.ds)}"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, k):
        raw = self.ds[k]["raw_emg"].float()               # (T, 8)
        logits = np.load(f"{self.cache_dir}/{k}.npy")      # (T', 38) fp16
        logits = torch.from_numpy(logits).float()          # -> fp32
        label = torch.from_numpy(self.labels[k].astype(np.int64))  # (L,)
        return raw, logits, label


def kd_collate(batch):
    """Pad a batch to equal length; return tensors + real lengths."""
    raws, logits, labels = zip(*batch)

    raw_lens   = torch.tensor([r.shape[0] for r in raws], dtype=torch.long)
    logit_lens = torch.tensor([l.shape[0] for l in logits], dtype=torch.long)  # student frame counts
    label_lens = torch.tensor([y.shape[0] for y in labels], dtype=torch.long)

    raw_pad    = pad_sequence(raws,   batch_first=True)     # (B, Tmax, 8)
    logit_pad  = pad_sequence(logits, batch_first=True)     # (B, T'max, 38)
    labels_cat = torch.cat(labels)                          # (sum L,) flat for CTC

    return raw_pad, logit_pad, labels_cat, raw_lens, logit_lens, label_lens