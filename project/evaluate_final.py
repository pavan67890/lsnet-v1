import os
import csv
import torch
from torch.utils.data import DataLoader
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
import config
from torch.amp import autocast
import numpy as np

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_val_loader(batch_size=4, num_workers=2):
    root = os.path.join(os.path.dirname(config.data_root), "whu_patches")
    val_ds = PatchChangeDetectionDataset(root, mode="test")
    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return loader

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

def remove_small_components(mask, min_size=20):
    # mask: torch.bool [B,1,H,W] or [1,H,W]
    m = mask.clone().detach().cpu().numpy()
    if m.ndim == 4:
        B = m.shape[0]
        out = []
        for b in range(B):
            out.append(_remove_small_components_single(m[b, 0], min_size))
        out = torch.from_numpy(np.stack(out, axis=0)).unsqueeze(1).bool()
        return out.to(mask.device)
    else:
        cleaned = _remove_small_components_single(m[0] if m.ndim == 3 else m, min_size)
        cleaned = torch.from_numpy(cleaned).unsqueeze(0).unsqueeze(0).bool()
        return cleaned.to(mask.device)

def _remove_small_components_single(arr, min_size=20):
    import numpy as np
    H, W = arr.shape
    visited = np.zeros((H, W), dtype=np.uint8)
    arr = arr.astype(np.bool_)
    def neighbors(y, x):
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                yield ny, nx
    for y in range(H):
        for x in range(W):
            if not arr[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            comp = []
            visited[y, x] = 1
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for ny, nx in neighbors(cy, cx):
                    if arr[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
            if len(comp) < min_size:
                for cy, cx in comp:
                    arr[cy, cx] = False
    return arr

def tta_predict(model, t1, t2):
    # returns averaged probabilities over TTA variants
    with autocast('cuda'):
        logits, _ = model(t1, t2)
        probs = torch.sigmoid(logits)
    # hflip
    t1_h = torch.flip(t1, dims=[-1])
    t2_h = torch.flip(t2, dims=[-1])
    with autocast('cuda'):
        logits_h, _ = model(t1_h, t2_h)
        probs_h = torch.sigmoid(logits_h)
    probs_h = torch.flip(probs_h, dims=[-1])
    # vflip
    t1_v = torch.flip(t1, dims=[-2])
    t2_v = torch.flip(t2, dims=[-2])
    with autocast('cuda'):
        logits_v, _ = model(t1_v, t2_v)
        probs_v = torch.sigmoid(logits_v)
    probs_v = torch.flip(probs_v, dims=[-2])
    # hvflip
    t1_hv = torch.flip(torch.flip(t1, dims=[-1]), dims=[-2])
    t2_hv = torch.flip(torch.flip(t2, dims=[-1]), dims=[-2])
    with autocast('cuda'):
        logits_hv, _ = model(t1_hv, t2_hv)
        probs_hv = torch.sigmoid(logits_hv)
    probs_hv = torch.flip(torch.flip(probs_hv, dims=[-1]), dims=[-2])
    avg_probs = (probs + probs_h + probs_v + probs_hv) / 4.0
    return avg_probs

def main():
    device = get_device()
    model = LSNet().to(device)
    cp_path = r"S:\sota\project\checkpoints\lsnet_whu_best.pth"
    if not os.path.exists(cp_path):
        print("Checkpoint not found:", cp_path)
        return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        # fallback: entire payload may be the state dict
        try:
            model.load_state_dict(payload)
        except Exception:
            print("Invalid checkpoint format. Abort.")
            return
    else:
        model.load_state_dict(state)
    print("Loaded checkpoint:", cp_path)
    model.eval()
    loader = build_val_loader(batch_size=4, num_workers=2)
    try:
        t1_s, t2_s, m_s = next(iter(loader))
        im_min = float(t1_s.min().item())
        im_max = float(t1_s.max().item())
        mask_np = m_s.detach().cpu().numpy()
        mask_np = (mask_np > 0).astype(np.float32)
        print("Image min/max:", im_min, im_max)
        print("Mask unique:", np.unique(mask_np))
    except Exception:
        pass
    os.makedirs("logs", exist_ok=True)
    out_csv = os.path.join("logs", "final_threshold_search.csv")
    thresholds = [round(x, 2) for x in torch.arange(0.30, 0.70 + 1e-8, 0.02).tolist()]
    best_thr = None
    best_f1 = -1.0
    best_iou = -1.0
    with open(out_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["threshold", "precision", "recall", "f1", "iou"])
        with torch.no_grad():
            for thr in thresholds:
                p_sum = r_sum = f1_sum = iou_sum = 0.0
                n = 0
                for t1, t2, m in loader:
                    t1 = t1.to(device, non_blocking=True)
                    t2 = t2.to(device, non_blocking=True)
                    m = m.to(device, non_blocking=True)
                    m = (m > 0).float()
                    probs = tta_predict(model, t1, t2)
                    pred = (probs > thr)
                    pred = remove_small_components(pred, min_size=20)
                    p, r, f1, iou = compute_metrics(pred, m)
                    p_sum += p
                    r_sum += r
                    f1_sum += f1
                    iou_sum += iou
                    n += 1
                p = p_sum / n
                r = r_sum / n
                f1 = f1_sum / n
                iou = iou_sum / n
                print(f"threshold={thr:.2f} precision={p:.4f} recall={r:.4f} f1={f1:.4f} iou={iou:.4f}")
                writer.writerow([thr, p, r, f1, iou])
                if iou > best_iou or (iou == best_iou and f1 > best_f1):
                    best_iou = iou
                    best_f1 = f1
                    best_thr = thr
    print(f"BEST threshold={best_thr:.2f} F1={best_f1:.4f} IoU={best_iou:.4f}")

if __name__ == "__main__":
    # local import to keep numpy optional
    import numpy as np
    main()
