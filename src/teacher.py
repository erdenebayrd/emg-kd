%%writefile /content/emg-kd/src/teacher.py
import sys, torch
from absl import flags

def load_teacher(ckpt_path, device="cuda"):
    """Load frozen Gaddy 2021 teacher. Returns model in eval mode."""
    FLAGS = flags.FLAGS
    if not FLAGS.is_parsed():
        FLAGS([sys.argv[0]])

    from architecture import Model
    model = Model(num_features=112, num_outs=38)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model