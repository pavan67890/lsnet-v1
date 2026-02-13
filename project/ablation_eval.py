import os
import numpy as np
import torch
from torch.amp import autocast
from contextlib import nullcontext
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.lsnet import LSNet
import train as train_mod
from utils import metrics

def tta_predict(model, t1, t2, device):
    ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
    with ctx:
        logits, _ = model(t1, t2)
        probs = torch.sigmoid(logits)
    t1_h = torch.flip(t1, dims=[-1]); t2_h = torch.flip(t2, dims=[-1])
    ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
    with ctx:
        lh, _ = model(t1_h, t2_h)
        ph = torch.sigmoid(lh)
    ph = torch.flip(ph, dims=[-1])
    t1_v = torch.flip(t1, dims=[-2]); t2_v = torch.flip(t2, dims=[-2])
    ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
    with ctx:
        lv, _ = model(t1_v, t2_v)
        pv = torch.sigmoid(lv)
    pv = torch.flip(pv, dims=[-2])
    t1_hv = torch.flip(torch.flip(t1, dims=[-1]), dims=[-2]); t2_hv = torch.flip(torch.flip(t2, dims=[-1]), dims=[-2])
    ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
    with ctx:
        lhv, _ = model(t1_hv, t2_hv)
        phv = torch.sigmoid(lhv)
    phv = torch.flip(torch.flip(phv, dims=[-1]), dims=[-2])
    return (probs + ph + pv + phv) / 4.0

def eval_once(model, val_loader, device, threshold, use_tta):
    p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
    with torch.no_grad():
        for t1, t2, m in val_loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            if use_tta:
                probs = tta_predict(model, t1, t2, device)
            else:
                ctx = autocast('cuda') if device.type == 'cuda' else nullcontext()
                with ctx:
                    logits, _ = model(t1, t2)
                    probs = torch.sigmoid(logits)
            p, r, f1, iou = metrics(probs, m, threshold=threshold)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
    return p_sum/n, r_sum/n, f1_sum/n, iou_sum/n

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    cp_path = r"S:\sota\project\checkpoints\lsnet_whu_best.pth"
    if not os.path.exists(cp_path):
        print("Checkpoint not found:", cp_path); return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        try:
            model.load_state_dict(payload)
        except Exception:
            print("Invalid checkpoint format. Abort."); return
    else:
        model.load_state_dict(state)
    print("Loaded checkpoint:", cp_path)
    model.eval()
    _, val_loader = train_mod.build_dataloaders()
    base = r"S:\sota\project\paper_outputs"
    os.makedirs(base, exist_ok=True)
    try:
        with open(os.path.join(base, "ablation_touch.txt"), "w") as f:
            f.write("touch\n")
    except Exception:
        pass
    # Baseline: 0.5, no TTA
    _, _, f1_base, iou_base = eval_once(model, val_loader, device, threshold=0.5, use_tta=False)
    # Threshold tuning: 0.34, no TTA
    _, _, f1_thr, iou_thr = eval_once(model, val_loader, device, threshold=0.34, use_tta=False)
    # TTA final: 0.34, TTA
    _, _, f1_tta, iou_tta = eval_once(model, val_loader, device, threshold=0.34, use_tta=True)
    labels = ["Baseline(0.5)", "Threshold(0.34)", "TTA(final)"]
    f1_vals = [f1_base, f1_thr, f1_tta]; iou_vals = [iou_base, iou_thr, iou_tta]
    x = np.arange(len(labels)); width = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, f1_vals, width, label="F1")
    plt.bar(x + width/2, iou_vals, width, label="IoU")
    plt.xticks(x, labels)
    plt.ylabel("Score"); plt.title("Ablation Study")
    for i, v in enumerate(f1_vals):
        plt.text(i - width/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(iou_vals):
        plt.text(i + width/2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.legend(); plt.tight_layout()
    out_path = os.path.join(base, "ablation_study.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved:", out_path)
    try:
        with open(os.path.join(base, "ablation_debug.txt"), "w") as f:
            f.write(f"F1: {f1_vals}\n")
            f.write(f"IoU: {iou_vals}\n")
            f.write(f"Out: {out_path}\n")
    except Exception:
        pass

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    with torch.no_grad():
        main()
