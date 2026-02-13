import os
import csv
import math
import torch
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader, Subset
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
from utils.utils import metrics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from train_levir import build_loaders
import numpy as np

def tta_predict(model, t1, t2, device):
    ctx = autocast('cuda') if device.type=='cuda' else nullcontext()
    with ctx:
        logits, _ = model(t1, t2)
        probs = torch.sigmoid(logits)
    t1h = torch.flip(t1, [-1]); t2h = torch.flip(t2, [-1])
    with ctx:
        lh, _ = model(t1h, t2h)
        ph = torch.sigmoid(lh)
    ph = torch.flip(ph, [-1])
    t1v = torch.flip(t1, [-2]); t2v = torch.flip(t2, [-2])
    with ctx:
        lv, _ = model(t1v, t2v)
        pv = torch.sigmoid(lv)
    pv = torch.flip(pv, [-2])
    t1hv = torch.flip(torch.flip(t1, [-1]), [-2]); t2hv = torch.flip(torch.flip(t2, [-1]), [-2])
    with ctx:
        lhv, _ = model(t1hv, t2hv)
        phv = torch.sigmoid(lhv)
    phv = torch.flip(torch.flip(phv, [-1]), [-2])
    return (probs + ph + pv + phv) / 4.0

def tta_predict_6way(model, t1, t2, device):
    ctx = autocast('cuda') if device.type=='cuda' else nullcontext()
    outs = []
    with ctx:
        l0, _ = model(t1, t2)
        p0 = torch.sigmoid(l0); outs.append(p0)
    t1h = torch.flip(t1, [-1]); t2h = torch.flip(t2, [-1])
    with ctx:
        lh, _ = model(t1h, t2h)
        ph = torch.sigmoid(lh); ph = torch.flip(ph, [-1]); outs.append(ph)
    t1v = torch.flip(t1, [-2]); t2v = torch.flip(t2, [-2])
    with ctx:
        lv, _ = model(t1v, t2v)
        pv = torch.sigmoid(lv); pv = torch.flip(pv, [-2]); outs.append(pv)
    t1hv = torch.flip(torch.flip(t1, [-1]), [-2]); t2hv = torch.flip(torch.flip(t2, [-1]), [-2])
    with ctx:
        lhv, _ = model(t1hv, t2hv)
        phv = torch.sigmoid(lhv); phv = torch.flip(torch.flip(phv, [-1]), [-2]); outs.append(phv)
    t190 = torch.rot90(t1, 1, [-2, -1]); t290 = torch.rot90(t2, 1, [-2, -1])
    with ctx:
        l90, _ = model(t190, t290)
        p90 = torch.sigmoid(l90); p90 = torch.rot90(p90, 3, [-2, -1]); outs.append(p90)
    t1270 = torch.rot90(t1, 3, [-2, -1]); t2270 = torch.rot90(t2, 3, [-2, -1])
    with ctx:
        l270, _ = model(t1270, t2270)
        p270 = torch.sigmoid(l270); p270 = torch.rot90(p270, 1, [-2, -1]); outs.append(p270)
    return sum(outs) / float(len(outs))

def _make_loader(split, limit=None):
    root = os.path.join(config.PATCHES_ROOT, split)
    assert os.path.isdir(root)
    assert os.path.isdir(os.path.join(root, "mask"))
    ds = PatchChangeDetectionDataset(root, mode="test")
    base_ds = ds
    n = len(base_ds)
    if limit is not None and n > limit:
        ds = Subset(base_ds, list(range(limit)))
    return DataLoader(ds, batch_size=config.micro_batch_size, shuffle=False, num_workers=0, pin_memory=True)

def _load_model(device):
    model = LSNet().to(device)
    cp = os.path.join(config.CHECKPOINTS_DIR, "levir_best.pth")
    payload = torch.load(cp, map_location=device)
    state = payload.get("model", None)
    if state is None and isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    if state is None:
        state = payload
    model.load_state_dict(state)
    model.eval()
    epoch = None
    best_iou = None
    if isinstance(payload, dict):
        epoch = payload.get("epoch", payload.get("last_epoch", None))
        best_iou = payload.get("best_iou", None)
    print("loaded_checkpoint", cp)
    print("checkpoint_epoch", epoch)
    print("checkpoint_best_iou", best_iou)
    return model

def threshold_search_val(model, device):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    thrs = [round(x, 3) for x in torch.arange(0.80, 0.95 + 1e-8, 0.002).tolist()]
    out_path = os.path.join(config.LOGS_DIR, "final_threshold_search.csv")
    best_thr = None
    best_iou = -1.0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold","precision","recall","f1","iou"])
    with torch.no_grad():
        for thr in thrs:
            loader = _make_loader("val", limit=None)
            p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
            for t1, t2, m in loader:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                logits, _ = model(t1, t2)
                probs = torch.sigmoid(logits)
                p,r,f1,iou = metrics(probs, m, threshold=thr)
                p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
            precision = p_sum/n; recall = r_sum/n; f1 = f1_sum/n; iou = iou_sum/n
            with open(out_path, "a", newline="") as fa:
                wa = csv.writer(fa)
                wa.writerow([thr, precision, recall, f1, iou])
            if iou > best_iou:
                best_iou = iou; best_thr = thr
    print("Best threshold:", best_thr, "Best IoU:", best_iou)
    return best_thr, best_iou

def eval_val_normal_and_tta(model, device, thr):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    loader = _make_loader("val", limit=None)
    out_path = os.path.join(config.LOGS_DIR, "tta_val_results.csv")
    with torch.no_grad():
        p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
        for t1, t2, m in loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            logits, _ = model(t1, t2)
            probs = torch.sigmoid(logits)
            p,r,f1,iou = metrics(probs, m, threshold=thr)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
        precision_n = p_sum/n; recall_n = r_sum/n; f1_n = f1_sum/n; iou_n = iou_sum/n
        p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
        for t1, t2, m in loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict_6way(model, t1, t2, device)
            p,r,f1,iou = metrics(probs, m, threshold=thr)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
        precision_t = p_sum/n; recall_t = r_sum/n; f1_t = f1_sum/n; iou_t = iou_sum/n
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","precision","recall","f1","iou"])
        w.writerow(["normal", precision_n, recall_n, f1_n, iou_n])
        w.writerow(["tta6", precision_t, recall_t, f1_t, iou_t])
    print("Validation IoU normal:", iou_n, "Validation IoU TTA:", iou_t)
    return (precision_n, recall_n, f1_n, iou_n), (precision_t, recall_t, f1_t, iou_t)

def eval_test_tta(model, device, thr):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    loader = _make_loader("test", limit=None)
    out_path = os.path.join(config.LOGS_DIR, "final_test_metrics.csv")
    with torch.no_grad():
        p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
        for t1, t2, m in loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict_6way(model, t1, t2, device)
            p,r,f1,iou = metrics(probs, m, threshold=thr)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
        precision = p_sum/n; recall = r_sum/n; f1 = f1_sum/n; iou = iou_sum/n
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision","recall","f1","iou"])
        w.writerow([precision, recall, f1, iou])
    return precision, recall, f1, iou

def global_threshold_search_narrow(model, device):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    _, vloader = build_loaders(max_val_patches=None)
    th = torch.arange(0.85, 0.95 + 1e-8, 0.002, device=device)
    tp = torch.zeros_like(th, dtype=torch.float64)
    fp = torch.zeros_like(th, dtype=torch.float64)
    fn = torch.zeros_like(th, dtype=torch.float64)
    with torch.no_grad():
        for t1, t2, m in vloader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            logits, probs = model(t1, t2)
            if m.max().item() > 1.0:
                m = m / 255.0
            pos = (m > 0.5)
            neg = ~pos
            for i in range(th.numel()):
                pred = probs > th[i]
                tp[i] += (pred & pos).sum()
                fp[i] += (pred & neg).sum()
                fn[i] += ((~pred) & pos).sum()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    best_idx = int(torch.argmax(iou).item())
    best_thr = float(th[best_idx].item())
    best_iou = float(iou[best_idx].item())
    out = os.path.join(config.LOGS_DIR, "final_threshold_search.csv")
    lines = ["threshold,precision,recall,f1,iou"]
    for i in range(th.numel()):
        lines.append(f"{float(th[i].item())},{float(precision[i].item())},{float(recall[i].item())},{float(f1[i].item())},{float(iou[i].item())}")
    with open(out, "w") as f:
        s = "\n".join(lines)
        f.write(s)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    print("Best threshold", best_thr, "Best IoU", best_iou)
    return best_thr, best_iou

def eval_val_tta_4way_global(model, device, thr):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    _, vloader = build_loaders(max_val_patches=None)
    tp_n = fp_n = fn_n = 0.0
    tp_t = fp_t = fn_t = 0.0
    with torch.no_grad():
        for t1, t2, m in vloader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            logits, probs = model(t1, t2)
            if m.max().item() > 1.0:
                m = m / 255.0
            gt = (m > 0.5)
            pred_n = (probs > thr)
            tp_n += (pred_n & gt).sum().item(); fp_n += (pred_n & (~gt)).sum().item(); fn_n += ((~pred_n) & gt).sum().item()
            t1h = torch.flip(t1, [-1]); t2h = torch.flip(t2, [-1]); lh, ph = model(t1h, t2h); ph = torch.flip(ph, [-1])
            t1v = torch.flip(t1, [-2]); t2v = torch.flip(t2, [-2]); lv, pv = model(t1v, t2v); pv = torch.flip(pv, [-2])
            t1hv = torch.flip(torch.flip(t1, [-1]), [-2]); t2hv = torch.flip(torch.flip(t2, [-1]), [-2]); lhv, phv = model(t1hv, t2hv); phv = torch.flip(torch.flip(phv, [-1]), [-2])
            pavg = (probs + ph + pv + phv) / 4.0
            pred_t = (pavg > thr)
            tp_t += (pred_t & gt).sum().item(); fp_t += (pred_t & (~gt)).sum().item(); fn_t += ((~pred_t) & gt).sum().item()
    eps = 1e-8
    prec_n = tp_n/(tp_n+fp_n+eps); rec_n = tp_n/(tp_n+fn_n+eps); f1_n = 2*prec_n*rec_n/(prec_n+rec_n+eps); iou_n = tp_n/(tp_n+fp_n+fn_n+eps)
    prec_t = tp_t/(tp_t+fp_t+eps); rec_t = tp_t/(tp_t+fn_t+eps); f1_t = 2*prec_t*rec_t/(prec_t+rec_t+eps); iou_t = tp_t/(tp_t+fp_t+fn_t+eps)
    out = os.path.join(config.LOGS_DIR, "final_tta_metrics.txt")
    text = "\n".join([
        "TTA Evaluation Results",
        f"Threshold used: {thr}",
        f"Normal: Precision {prec_n} Recall {rec_n} F1 {f1_n} IoU {iou_n}",
        f"TTA: Precision {prec_t} Recall {rec_t} F1 {f1_t} IoU {iou_t}",
        f"IoU gain {iou_t - iou_n}",
        f"F1 gain {f1_t - f1_n}",
    ])
    with open(out, "w") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    return (prec_n, rec_n, f1_n, iou_n), (prec_t, rec_t, f1_t, iou_t)

def visuals_from_search_and_metrics(best_thr, normal, tta):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(config.LOGS_DIR, "final_threshold_search.csv")
    th = []; iou = []; pr_x = []; pr_y = []
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            rows = [line.strip() for line in f.readlines() if line.strip()]
        for line in rows[1:]:
            parts = line.split(",")
            th.append(float(parts[0])); pr_x.append(float(parts[1])); pr_y.append(float(parts[2])); iou.append(float(parts[4]))
    if len(th) > 0:
        plt.figure(figsize=(6,5)); plt.plot(th, iou); plt.xlabel("Threshold"); plt.ylabel("IoU"); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(config.LOGS_DIR, "threshold_vs_iou.png"), dpi=300); plt.close()
        plt.figure(figsize=(6,5)); plt.plot(pr_y, pr_x); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(config.LOGS_DIR, "precision_recall_curve.png"), dpi=300); plt.close()
    _, vloader = build_loaders(max_val_patches=None)
    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for t1, t2, m in vloader:
            t1 = t1.to(next(LSNet().parameters()).device)  # dummy to satisfy device, not used
            m = m
            gt = (m > 0.5)
            tn += ((~gt) & (~gt)).sum().item()
    n_prec_n, n_rec_n, n_f1_n, n_iou_n = normal
    n_prec_t, n_rec_t, n_f1_t, n_iou_t = tta
    plt.figure(figsize=(6,4))
    labels = ["Normal", "TTA"]
    ious = [n_iou_n, n_iou_t]
    f1s = [n_f1_n, n_f1_t]
    x = np.arange(len(labels))
    w = 0.35
    plt.bar(x - w/2, ious, width=w, label="IoU")
    plt.bar(x + w/2, f1s, width=w, label="F1")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOGS_DIR, "normal_vs_tta_bar.png"), dpi=300)
    plt.close()

def resume_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)
    vloader = _make_loader("val", limit=None)
    tloader = _make_loader("test", limit=None)
    print("device", device.type)
    print("val_len", len(vloader.dataset))
    print("test_len", len(tloader.dataset))
    with torch.no_grad():
        t1, t2, m = next(iter(vloader))
        print("val_batch_shapes", t1.shape, t2.shape, m.shape)
        print("val_mask_min_max", m.min().item(), m.max().item())
    return device, model, vloader

def sanity_checks(model, device):
    loader = _make_loader("val", limit=None)
    with torch.no_grad():
        t1, t2, m = next(iter(loader))
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        logits, _ = model(t1, t2)
        probs = torch.sigmoid(logits)
        print("probs_mean", probs.mean().item())
        print("probs_min", probs.min().item())
        print("probs_max", probs.max().item())
        print("mask_min", m.min().item())
        print("mask_max", m.max().item())

def curves_and_visuals(model, device, thr):
    os.makedirs(os.path.join("S:/sota/levir_project", "paper_outputs"), exist_ok=True)
    out_dir = os.path.join("S:/sota/levir_project", "paper_outputs")
    loader = _make_loader("val", limit=None)
    thrs = [round(x, 3) for x in torch.arange(0.80, 0.95 + 1e-8, 0.002).tolist()]
    pr = []; roc = []; f1s = []; ious = []
    with torch.no_grad():
        for t in thrs:
            p_sum = r_sum = f1_sum = iou_sum = 0.0; n=0
            tp_sum = fp_sum = fn_sum = tn_sum = 0.0
            for t1, t2, m in loader:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                probs = tta_predict_6way(model, t1, t2, device)
                mm = m
                if mm.max().item() > 1.0:
                    mm = mm / 255.0
                p,r,f1,iou = metrics(probs, mm, threshold=t)
                p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n+=1
                preds = (probs > t).float()
                targets = (mm > 0.5).float()
                tp_sum += (preds * targets).sum().item()
                fp_sum += (preds * (1-targets)).sum().item()
                fn_sum += ((1-preds) * targets).sum().item()
                tn_sum += ((1-preds) * (1-targets)).sum().item()
            pr.append((p_sum/n, r_sum/n))
            roc.append((tp_sum/(tp_sum+fn_sum+1e-8), fp_sum/(fp_sum+tn_sum+1e-8)))
            f1s.append(f1_sum/n); ious.append(iou_sum/n)
    plt.figure(figsize=(6,5))
    plt.plot([x for x,_ in pr], [y for _,y in pr], label="PR")
    plt.xlabel("Precision"); plt.ylabel("Recall")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "precision_recall.png"), dpi=300)
    plt.figure(figsize=(6,5))
    plt.plot([x for x,_ in roc], [y for _,y in roc], label="ROC")
    plt.xlabel("TPR"); plt.ylabel("FPR")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300)
    plt.figure(figsize=(6,5))
    plt.plot(thrs, ious); plt.xlabel("Threshold"); plt.ylabel("IoU")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "thr_vs_iou.png"), dpi=300)
    plt.figure(figsize=(6,5))
    plt.plot(thrs, f1s); plt.xlabel("Threshold"); plt.ylabel("F1")
    plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "thr_vs_f1.png"), dpi=300)
    loader = _make_loader("val", limit=None)
    k = 0
    with torch.no_grad():
        for t1, t2, m in loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict_6way(model, t1, t2, device)
            mm = m
            if mm.max().item() > 1.0:
                mm = mm / 255.0
            pred = (probs > thr).float()
            B = t1.size(0)
            for i in range(B):
                if k >= 10: break
                fig, axs = plt.subplots(1,4,figsize=(10,3))
                axs[0].imshow(t1[i].permute(1,2,0).cpu().numpy()); axs[0].axis("off"); axs[0].set_title("T1")
                axs[1].imshow(t2[i].permute(1,2,0).cpu().numpy()); axs[1].axis("off"); axs[1].set_title("T2")
                axs[2].imshow(mm[i,0].cpu().numpy(), cmap="gray"); axs[2].axis("off"); axs[2].set_title("GT")
                axs[3].imshow(pred[i,0].cpu().numpy(), cmap="gray"); axs[3].axis("off"); axs[3].set_title("Pred")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"qual_{k+1:02d}.png"), dpi=300)
                plt.close(fig)
                k += 1
            if k >= 10: break

def threshold_search_val_range(model, device, start, stop, step, out_name):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    thrs = [round(x, 3) for x in torch.arange(start, stop + 1e-8, step).tolist()]
    out_path = os.path.join(config.LOGS_DIR, out_name)
    best_thr = None
    best_iou = -1.0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold","precision","recall","f1","iou"])
    with torch.no_grad():
        for thr in thrs:
            loader = _make_loader("val", limit=None)
            p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
            for t1, t2, m in loader:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                logits, _ = model(t1, t2)
                probs = torch.sigmoid(logits)
                mm = m
                if mm.max().item() > 1.0:
                    mm = mm / 255.0
                p,r,f1,iou = metrics(probs, mm, threshold=thr)
                p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
            precision = p_sum/n; recall = r_sum/n; f1 = f1_sum/n; iou = iou_sum/n
            with open(out_path, "a", newline="") as fa:
                wa = csv.writer(fa)
                wa.writerow([thr, precision, recall, f1, iou])
            if iou > best_iou:
                best_iou = iou; best_thr = thr
            print("done_thr", thr, "iou", iou)
    print("Best threshold", best_thr)
    print("Best IoU", best_iou)
    return best_thr, best_iou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = _load_model(device)
        best_thr, best_iou_val = global_threshold_search_narrow(model, device)
        normal, tta = eval_val_tta_4way_global(model, device, best_thr)
        visuals_from_search_and_metrics(best_thr, normal, tta)
    except RuntimeError as e:
        if "CUDA" in str(e) or "cublas" in str(e):
            device = torch.device("cpu")
            model = _load_model(device)
            best_thr, best_iou_val = global_threshold_search_narrow(model, device)
            normal, tta = eval_val_tta_4way_global(model, device, best_thr)
            visuals_from_search_and_metrics(best_thr, normal, tta)
        else:
            raise

if __name__ == "__main__":
    with torch.no_grad():
        main()
