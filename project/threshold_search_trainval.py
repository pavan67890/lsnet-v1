import os
import torch
import csv
from torch.amp import autocast
from models.lsnet import LSNet
import train as train_mod
from utils import metrics

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
    thresholds = [round(x, 2) for x in torch.arange(0.20, 0.30 + 1e-8, 0.02).tolist()]
    print("threshold | precision | recall | f1 | iou")
    best_f1 = -1.0
    best_f1_thr = None
    best_iou = -1.0
    best_iou_thr = None
    os.makedirs("logs", exist_ok=True)
    out_csv = os.path.join("logs", "trainval_thresholds.csv")
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["threshold", "precision", "recall", "f1", "iou"])
        with torch.no_grad():
            for thr in thresholds:
                p_sum = r_sum = f1_sum = iou_sum = 0.0
                n = 0
                for t1, t2, m in val_loader:
                    t1 = t1.to(device, non_blocking=True)
                    t2 = t2.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)
                    with autocast('cuda'):
                        logits, probs = model(t1, t2)
                    p, r, f1, iou = metrics(probs, m, threshold=thr)
                    p_sum += p
                    r_sum += r
                    f1_sum += f1
                    iou_sum += iou
                    n += 1
                p = p_sum / n
                r = r_sum / n
                f1 = f1_sum / n
                iou = iou_sum / n
                print(f"{thr:.2f} | {p:.4f} | {r:.4f} | {f1:.4f} | {iou:.4f}")
                writer.writerow([thr, p, r, f1, iou])
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_thr = thr
                if iou > best_iou:
                    best_iou = iou
                    best_iou_thr = thr
    print("BEST_F1 threshold", best_f1_thr, "F1", round(best_f1, 4))
    print("BEST_IOU threshold", best_iou_thr, "IoU", round(best_iou, 4))

if __name__ == "__main__":
    main()
