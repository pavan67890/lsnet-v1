import time
import torch
import torch.nn as nn
from torch.amp import autocast
from contextlib import nullcontext
from models.lsnet import LSNet
import config
from utils import count_params

def conv_flops(module, inp, out):
    x = inp[0]
    B, Cin, H, W = x.shape
    Cout = module.out_channels
    kh, kw = module.kernel_size
    groups = module.groups
    # per-output-pixel MACs = (Cin/groups)*kh*kw
    macs_per_out = (Cin // groups) * kh * kw
    # total MACs = B * Cout * Hout * Wout * macs_per_out
    Hout, Wout = out.shape[-2], out.shape[-1]
    macs = B * Cout * Hout * Wout * macs_per_out
    # FLOPs ~ 2 * MACs (multiply + add)
    flops = 2 * macs
    return flops

def profile_gflops(model, device, bs=1, size=None):
    if size is None:
        size = int(getattr(config, "patch_size", 256))
    t1 = torch.randn(bs, 3, size, size, device=device)
    t2 = torch.randn(bs, 3, size, size, device=device)
    flops_total = 0
    hooks = []
    # helper closure
    def nonlocal_add(mod, inp, out):
        nonlocal flops_total
        flops_total += conv_flops(mod, inp, out)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(lambda mod, inp, out: nonlocal_add(mod, inp, out)))
    # warmup once to initialize shapes
    model.eval()
    with torch.no_grad():
        ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
        with ctx:
            _ = model(t1, t2)
    for h in hooks:
        h.remove()
    gflops = flops_total / 1e9
    return gflops

def measure_fps(model, device, bs=4, size=None, iters=50, warmup=10):
    if size is None:
        size = int(getattr(config, "patch_size", 256))
    t1 = torch.randn(bs, 3, size, size, device=device)
    t2 = torch.randn(bs, 3, size, size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
            with ctx:
                _ = model(t1, t2)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(iters):
            ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
            with ctx:
                _ = model(t1, t2)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1_time = time.time() - t0
    fps = (iters * bs) / t1_time
    return fps

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    params = count_params(model)
    gflops_1 = profile_gflops(model, device, bs=1)
    fps_bs4 = measure_fps(model, device, bs=4)
    print("params", params)
    print("gflops_per_forward_bs1", round(gflops_1, 3))
    print("fps_bs4", round(fps_bs4, 2))
    try:
        import os
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "model_profile.txt"), "w") as f:
            f.write(f"params {params}\n")
            f.write(f"gflops_per_forward_bs1 {round(gflops_1, 3)}\n")
            f.write(f"fps_bs4 {round(fps_bs4, 2)}\n")
    except Exception:
        pass

if __name__ == "__main__":
    main()
