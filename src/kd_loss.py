# src/kd_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK = 37


def kd_ctc_loss(student_logits, teacher_logits, targets, in_lens, tgt_lens,
                alpha=0.5, temperature=2.0, blank=BLANK,
                mask_blank_dominant=True, blank_thresh=0.9):
    """
    Combined CTC + knowledge-distillation loss.

    student_logits : (B, T', V)  raw logits from student
    teacher_logits : (B, T', V)  raw logits from frozen teacher (same T', V)
    targets        : (sum(tgt_lens),) or (B, L)  flat/padded label ids
    in_lens        : (B,)  valid student frame counts (time steps)
    tgt_lens       : (B,)  label lengths
    alpha          : weight on the KD term; (1-alpha) on CTC
    temperature    : softmax temperature for the KD term

    returns: total_loss, {"ctc":..., "kd":...}
    """
    B, Tp, V = student_logits.shape

    # ---- CTC term: needs (T', B, V) log-probs ----
    log_probs = student_logits.log_softmax(-1).transpose(0, 1)
    ctc = F.ctc_loss(log_probs, targets, in_lens, tgt_lens,
                     blank=blank, zero_infinity=True)

    # ---- frame mask over padded time steps ----
    ar = torch.arange(Tp, device=student_logits.device)[None, :]
    frame_mask = (ar < in_lens[:, None]).float()          # (B, T')

    # ---- temperature-scaled KL(teacher || student) ----
    T = temperature
    s_log = (student_logits / T).log_softmax(-1)
    t_prob = (teacher_logits / T).softmax(-1)
    kl = F.kl_div(s_log, t_prob, reduction="none").sum(-1)   # (B, T')

    if mask_blank_dominant:
        # CTC posteriors are "peaky": most frames are ~100% blank, so naive KL
        # just teaches "predict blank". Down-weight teacher-blank-dominant frames.
        t_blank = t_prob[..., blank]
        frame_mask = frame_mask * (t_blank < blank_thresh).float()

    denom = frame_mask.sum().clamp(min=1.0)
    kd = (kl * frame_mask).sum() / denom * (T * T)    # T^2 keeps grad scale stable

    total = (1 - alpha) * ctc + alpha * kd
    return total, {"ctc": ctc.item(), "kd": kd.item()}