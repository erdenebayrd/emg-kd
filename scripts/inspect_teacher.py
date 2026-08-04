import torch, sys, collections

path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
except Exception as e:
    print("weights_only=True failed:", type(e).__name__, "- retrying")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) \
     if isinstance(ckpt, dict) else ckpt

print("top-level type:", type(ckpt).__name__)
if isinstance(ckpt, dict):
    first = next(iter(ckpt.values()), None)
    if not torch.is_tensor(first):
        print("top-level keys:", list(ckpt.keys())[:10])

print(f"\n{len(sd)} tensors | {sum(v.numel() for v in sd.values())/1e6:.2f}M params\n")

groups = collections.defaultdict(int)
for k, v in sd.items():
    groups[k.split('.')[0]] += v.numel()
print("by top-level module:")
for g, n in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"  {g:24s} {n/1e6:8.3f}M")

print("\nall keys and shapes:")
for k, v in sd.items():
    print(f"  {k:60s} {tuple(v.shape)}")