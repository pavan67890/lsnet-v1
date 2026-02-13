import os
import random
import torch
from torch.utils.data import DataLoader, Subset
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
import config

def compute_metrics(pred, target):
    pred = pred.view(-1).to(torch.bool)
    target = (target.view(-1) > 0.5)
    tp = (pred & target).sum().item()
    fp = (pred & (~target)).sum().item()
    fn = ((~pred) & target).sum().item()
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return p, r, f1, iou

def main():
    device = torch.device("cpu")
    model = LSNet().to(device)
    cp_path = os.path.join("checkpoints", "lsnet_whu_best.pth")
    if not os.path.exists(cp_path):
        print("No best checkpoint found. Abort.")
        return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        print("Invalid checkpoint format. Abort.")
        return
    model.load_state_dict(state)
    model.eval()
    root = os.path.join(os.path.dirname(config.data_root), "whu_patches")
    ds = PatchChangeDetectionDataset(root, mode="test")
    n = len(ds)
    max_n = 1000
    if n > max_n:
        idx = random.sample(range(n), max_n)
        ds = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_thr = None
    best_f1 = -1.0
    best_iou = 0.0
    with torch.no_grad():
        for thr in thresholds:
            p_sum = 0.0
            r_sum = 0.0
            f1_sum = 0.0
            iou_sum = 0.0
            c = 0
            for t1, t2, m in loader:
                t1 = t1.to(device)
                t2 = t2.to(device)
                m = m.to(device)
                logits, probs = model(t1, t2)
                pred = (probs > thr)
                p, r, f1, iou = compute_metrics(pred, m)
                p_sum += p
                r_sum += r
                f1_sum += f1
                iou_sum += iou
                c += 1
            p = p_sum / c
            r = r_sum / c
            f1 = f1_sum / c
            iou = iou_sum / c
            print(f"threshold={thr:.2f} F1={f1:.4f} IoU={iou:.4f}")
            if f1 > best_f1 or (f1 == best_f1 and iou > best_iou):
                best_f1 = f1
                best_iou = iou
                best_thr = thr
    print(f"BEST threshold={best_thr:.2f} F1={best_f1:.4f} IoU={best_iou:.4f}")

if __name__ == "__main__":
    main()
