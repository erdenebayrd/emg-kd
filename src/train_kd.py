# src/train_kd.py
import torch, time
from torch.utils.data import DataLoader
from src.kd_data import kd_collate
from src.kd_loss import kd_ctc_loss


def evaluate_wer(model, dev_dataset, chars, device, blank=37, max_n=None):
    """Greedy WER on a dev set, same decoder as the teacher baseline."""
    import jiwer
    model.eval()
    preds, refs = [], []
    n = len(dev_dataset) if max_n is None else min(max_n, len(dev_dataset))
    for k in range(n):
        item = dev_dataset[k]
        raw = item["raw_emg"].float().unsqueeze(0).to(device)
        with torch.no_grad():
            lg = model(raw)                       # (1, T', 38)
        ids = lg[0].argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != blank and i != prev:
                out.append(i)
            prev = i
        preds.append("".join(chars[i] for i in out))
        refs.append(item["text"])
    tfm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                         jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    rc = [tfm(r) for r in refs]; pc = [tfm(p) for p in preds]
    pairs = [(r, p) for r, p in zip(rc, pc) if r.strip()]
    rc, pc = [r for r, _ in pairs], [p for _, p in pairs]
    return jiwer.wer(rc, pc)


def train_student(student, kd_dataset, dev_dataset, chars, device,
                  epochs=10, batch_size=8, lr=3e-4,
                  alpha=0.5, temperature=2.0, eval_every=1, log_every=100,
                  ckpt_path=None):
    student.to(device)
    loader = DataLoader(kd_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=kd_collate, num_workers=4)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_wer = 1.0
    for ep in range(epochs):
        student.train()
        t0 = time.time()
        running = {"ctc": 0.0, "kd": 0.0}
        for step, (raw, t_logits, labels, rl, ll, yl) in enumerate(loader):
            raw      = raw.to(device)
            t_logits = t_logits.to(device)
            labels   = labels.to(device)
            ll, yl   = ll.to(device), yl.to(device)

            s_logits = student(raw)                     # (B, T', 38)

            # guard: trim to the shorter of student/teacher frame count per batch
            Tmin = min(s_logits.shape[1], t_logits.shape[1])
            s_logits = s_logits[:, :Tmin]
            t_logits = t_logits[:, :Tmin]
            ll = ll.clamp(max=Tmin)

            loss, parts = kd_ctc_loss(s_logits, t_logits, labels, ll, yl,
                                      alpha=alpha, temperature=temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()

            running["ctc"] += parts["ctc"]; running["kd"] += parts["kd"]
            if (step + 1) % log_every == 0:
                nb = step + 1
                print(f"  ep{ep} step{step+1}  ctc {running['ctc']/nb:.3f}  "
                      f"kd {running['kd']/nb:.3f}")
        sched.step()

        if (ep + 1) % eval_every == 0:
            wer = evaluate_wer(student, dev_dataset, chars, device)
            mark = ""

            if wer < best_wer:
                best_wer = wer; mark = "  <-- best"
                if ckpt_path:
                    save_ckpt(student, ckpt_path,
                              {"epoch": ep, "wer": wer,
                               "d_model": getattr(student, "_d_model", None)})

            print(f"epoch {ep} done in {time.time()-t0:.0f}s | "
                  f"dev WER {wer*100:.2f}%{mark}")
    return best_wer

def save_ckpt(student, path, meta=None):
    import torch
    torch.save({"model": student.state_dict(), "meta": meta or {}}, path)