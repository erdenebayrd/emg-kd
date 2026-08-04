%%writefile /content/emg-kd/src/teacher.py
import sys, torch
from absl import flags

TEACHER_VOCAB = 38      # 37 chars + CTC blank
TEACHER_BLANK = 37
TEACHER_DOWNSAMPLE = 8

def load_teacher(ckpt_path, device="cuda"):
    """Frozen Gaddy 2021 teacher. Call as: model(None, x_raw, None)
    x_raw is (batch, time, 8) raw EMG. Output is (batch, time/8, 38)."""
    FLAGS = flags.FLAGS
    if not FLAGS.is_parsed():
        FLAGS([sys.argv[0]])

    from architecture import Model
    model = Model(num_features=112, num_outs=38)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    # Gaddy's custom MultiHeadAttention predates this attribute;
    # modern nn.TransformerEncoder reads it. False = time-first.
    for layer in model.transformer.layers:
        layer.self_attn.batch_first = False

    model.eval().to(device)     # eval() disables the random time-shift augmentation
    for p in model.parameters():
        p.requires_grad_(False)
    return model