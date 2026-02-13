import os
import torch
import torch.nn.functional as F
from torch.amp import autocast
from models.lsnet import LSNet
import train as train_mod
from utils import metrics

def morph_open_close(mask, k=3):
    pad = k // 2
    x = mask.float()
    ones = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad) * (k * k)
    eroded = (ones == (k * k)).float()
    opened = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=pad)
    dilated = F.max_pool2d(opened, kernel_size=k, stride=1, padding=pad)
    ones2 = F.avg_pool2d(dilated, kernel_size=k, stride=1, padding=pad) * (k * k)
    closed = (ones2 == (k * k)).float()
    return closed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    cp_path = r"S:\sota\project\checkpoints\lsnet_whu_best.pth"
    if not os.path.exists(cp_path):
        print("Checkpoint not found:", cp_path)
        return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        try:
            model.load_state_dict(payload)
        except Exception:
            print("Invalid checkpoint format. Abort.")
            return
    else:
        model.load_state_dict(state)
    print("Loaded checkpoint:", cp_path)
    model.eval()
    _, val_loader = train_mod.build_dataloaders()
    thr = 0.22
    p_sum = r_sum = f1_sum = iou_sum = 0.0
    n = 0
    with torch.no_grad():
        for t1, t2, m in val_loader:
            t1 = t1.to(device, non_blocking=True)
            t2 = t2.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            with autocast('cuda'):
                logits, probs = model(t1, t2)
            pred = (probs > thr).float()
            pred = morph_open_close(pred, k=3)
            p, r, f1, iou = metrics(pred, m, threshold=0.5)
            p_sum += p
            r_sum += r
            f1_sum += f1
            iou_sum += iou
            n += 1
    print("precision", round(p_sum / n, 4))
    print("recall", round(r_sum / n, 4))
    print("f1", round(f1_sum / n, 4))
    print("iou", round(iou_sum / n, 4))

if __name__ == "__main__":
    main()
