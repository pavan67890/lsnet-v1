import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.amp import autocast
from models.lsnet import LSNet
import train as train_mod
from utils import metrics
from contextlib import nullcontext

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def ensure_dirs():
    base = r"S:\sota\project\paper_outputs"
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "qualitative_results"), exist_ok=True)
    return base

def final_evaluation(model, val_loader, device, base_dir, threshold=0.34):
    p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for t1, t2, m in val_loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict(model, t1, t2, device)
            p, r, f1, iou = metrics(probs, m, threshold=threshold)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
            pred = (probs > threshold).float()
            tgt = (m > 0.5).float()
            tp += (pred * tgt).sum().item()
            fp += (pred * (1 - tgt)).sum().item()
            fn += ((1 - pred) * tgt).sum().item()
            tn += ((1 - pred) * (1 - tgt)).sum().item()
    precision = p_sum / n; recall = r_sum / n; f1 = f1_sum / n; iou = iou_sum / n
    out_txt = os.path.join(base_dir, "final_metrics.txt")
    with open(out_txt, "w") as f:
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        f.write(f"IoU: {iou:.6f}\n")
    return precision, recall, f1, iou, tp, fp, fn, tn

def plot_confusion(tp, fp, fn, tn, base_dir):
    cm = np.array([[tp, fp],[fn, tn]], dtype=np.float64)
    norm = cm / (cm.sum() + 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, mat, title in zip(axes, [cm, norm], ["Counts", "Normalized"]):
        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0,1]); ax.set_xticklabels(["Pred=1","Pred=0"])
        ax.set_yticks([0,1]); ax.set_yticklabels(["True=1","True=0"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i,j]:.3f}" if title=="Normalized" else f"{int(mat[i,j])}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "confusion_matrix.png"), dpi=200)
    plt.close(fig)

def roc_pr_curves(model, val_loader, device, base_dir):
    thr = np.linspace(0.0, 1.0, 101)
    tpr_list = []; fpr_list = []; prec_list = []; rec_list = []
    with torch.no_grad():
        for t1, t2, m in val_loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict(model, t1, t2, device)
            probs_flat = probs.view(-1).detach().cpu().numpy()
            tgt_flat = (m.view(-1).detach().cpu().numpy() > 0.5).astype(np.uint8)
            for t in thr:
                pred = (probs_flat > t).astype(np.uint8)
                tp = np.sum((pred == 1) & (tgt_flat == 1)); fp = np.sum((pred == 1) & (tgt_flat == 0))
                fn = np.sum((pred == 0) & (tgt_flat == 1)); tn = np.sum((pred == 0) & (tgt_flat == 0))
                tpr = tp / (tp + fn + 1e-8); fpr = fp / (fp + tn + 1e-8)
                prec = tp / (tp + fp + 1e-8); rec = tpr
                tpr_list.append((t, tpr)); fpr_list.append((t, fpr)); prec_list.append((t, prec)); rec_list.append((t, rec))
    # Aggregate across batches by threshold
    def agg(seq):
        by_t = {}
        for t, v in seq:
            by_t.setdefault(t, []).append(v)
        xs = sorted(by_t.keys()); ys = np.array([np.mean(by_t[x]) for x in xs])
        return np.array(xs), ys
    xs_tpr, ys_tpr = agg(tpr_list); xs_fpr, ys_fpr = agg(fpr_list)
    order = np.argsort(xs_fpr)
    auc = np.trapezoid(ys_tpr[order], xs_fpr[order])
    plt.figure(figsize=(4,4)); plt.plot(ys_fpr, ys_tpr, label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],"--",color="gray"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "roc_curve.png"), dpi=200); plt.close()
    xs_prec, ys_prec = agg(prec_list); xs_rec, ys_rec = agg(rec_list)
    # Plot PR: recall vs precision
    plt.figure(figsize=(4,4)); plt.plot(ys_rec, ys_prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "pr_curve.png"), dpi=200); plt.close()
    return auc

def threshold_analysis(model, val_loader, device, base_dir):
    thrs = np.linspace(0.1, 0.9, 41)
    f1s = []; ious = []; precs = []; recs = []
    with torch.no_grad():
        for t in thrs:
            p_sum = r_sum = f1_sum = iou_sum = 0.0; n=0
            for t1, t2, m in val_loader:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                probs = tta_predict(model, t1, t2, device)
                p, r, f1, iou = metrics(probs, m, threshold=float(t))
                p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
            precs.append(p_sum/n); recs.append(r_sum/n); f1s.append(f1_sum/n); ious.append(iou_sum/n)
    plt.figure(figsize=(6,4)); plt.plot(thrs, f1s, label="F1"); plt.plot(thrs, ious, label="IoU"); plt.xlabel("Threshold"); plt.ylabel("Metric"); plt.title("Threshold Analysis"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "threshold_analysis.png"), dpi=200); plt.close()
    return thrs, f1s, ious, precs, recs

def qualitative_examples(model, val_loader, device, base_dir, threshold=0.34, count=10):
    # collect indices
    ds = val_loader.dataset
    n = len(ds)
    idxs = random.sample(range(n), min(count, n))
    out_dir = os.path.join(base_dir, "qualitative_results")
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            sample = ds[idx]
            if isinstance(sample, tuple):
                t1, t2, m = sample
            else:
                t1, t2, m = sample[0], sample[1], sample[2]
            t1b = t1.unsqueeze(0).to(device); t2b = t2.unsqueeze(0).to(device)
            probs = tta_predict(model, t1b, t2b, device)
            pred = (probs > threshold).float().squeeze(0).squeeze(0).detach().cpu().numpy()
            t1_np = np.clip(np.transpose(t1.detach().cpu().numpy(), (1,2,0)), 0.0, 1.0)
            t2_np = np.clip(np.transpose(t2.detach().cpu().numpy(), (1,2,0)), 0.0, 1.0)
            gt_np = (m.squeeze(0).detach().cpu().numpy() > 0.5).astype(np.uint8)
            fig, axes = plt.subplots(1,4, figsize=(10,3))
            axes[0].imshow(t1_np); axes[0].set_title("T1"); axes[0].axis("off")
            axes[1].imshow(t2_np); axes[1].set_title("T2"); axes[1].axis("off")
            axes[2].imshow(gt_np, cmap="gray"); axes[2].set_title("GT"); axes[2].axis("off")
            axes[3].imshow(pred, cmap="gray"); axes[3].set_title("Pred"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"sample_{i+1}.png"), dpi=200)
            plt.close(fig)

def training_curves(base_dir):
    log_path = os.path.join("logs", "whu_training_log.csv")
    epochs = []; losses = []; f1s = []; ious = []
    if not os.path.exists(log_path):
        return
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["epoch"]); l = float(row["train_loss"])
                f1 = float(row["f1"]) if row.get("f1") else None
                iou = float(row["iou"]) if row.get("iou") else None
                epochs.append(e); losses.append(l); f1s.append(f1); ious.append(iou)
            except Exception:
                pass
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.plot(epochs, losses); plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Train Loss")
    plt.subplot(1,2,2)
    ep_f1 = [e for e,f in zip(epochs, f1s) if f is not None]; val_f1 = [f for f in f1s if f is not None]
    ep_iou = [e for e,i in zip(epochs, ious) if i is not None]; val_iou = [i for i in ious if i is not None]
    if len(val_f1)>0: plt.plot(ep_f1, val_f1, label="F1")
    if len(val_iou)>0: plt.plot(ep_iou, val_iou, label="IoU")
    plt.xlabel("Epoch"); plt.ylabel("Metric"); plt.title("Validation Metrics"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(base_dir, "training_curves.png"), dpi=200); plt.close()

def ablation_study(model, val_loader, device, base_dir, final_thr=0.34):
    def eval_once(threshold, use_tta):
        p_sum = r_sum = f1_sum = iou_sum = 0.0; n = 0
        with torch.no_grad():
            for t1, t2, m in val_loader:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                if use_tta:
                    probs = tta_predict(model, t1, t2, device)
                else:
                    with autocast('cuda'):
                        logits, _ = model(t1, t2)
                        probs = torch.sigmoid(logits)
                p, r, f1, iou = metrics(probs, m, threshold=threshold)
                p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n += 1
        return p_sum/n, r_sum/n, f1_sum/n, iou_sum/n
    # Baseline: thr=0.5 no TTA
    _, _, f1_base, iou_base = eval_once(0.5, use_tta=False)
    # Threshold tuning: thr=final_thr no TTA
    _, _, f1_thr, iou_thr = eval_once(final_thr, use_tta=False)
    # Final TTA: thr=final_thr with TTA
    _, _, f1_tta, iou_tta = eval_once(final_thr, use_tta=True)
    labels = ["Baseline(0.5)", f"Threshold({final_thr})", "TTA(final)"]
    f1_vals = [f1_base, f1_thr, f1_tta]; iou_vals = [iou_base, iou_thr, iou_tta]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, f1_vals, width, label="F1")
    plt.bar(x + width/2, iou_vals, width, label="IoU")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Score"); plt.title("Ablation Study")
    # annotations disabled for robustness
    plt.legend(); plt.tight_layout()
    outp = os.path.join(base_dir, "ablation_study.png")
    plt.savefig(outp, dpi=200); plt.close()
    try:
        with open(os.path.join(base_dir, "ablation_debug.txt"), "w") as f:
            f.write(f"F1: {f1_vals}\n")
            f.write(f"IoU: {iou_vals}\n")
            f.write(f"Out: {outp}\n")
    except Exception:
        pass

def main():
    device = get_device()
    base = ensure_dirs()
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
    # 1. Final evaluation with TTA at threshold=0.34
    precision, recall, f1, iou, tp, fp, fn, tn = final_evaluation(model, val_loader, device, base, threshold=0.34)
    # 2. Confusion matrix
    plot_confusion(tp, fp, fn, tn, base)
    # 3. ROC and PR curves (with AUC)
    auc = roc_pr_curves(model, val_loader, device, base)
    # Append AUC to final metrics
    with open(os.path.join(base, "final_metrics.txt"), "a") as f:
        f.write(f"AUC: {auc:.6f}\n")
    # 5. Threshold vs metrics graphs (with TTA)
    threshold_analysis(model, val_loader, device, base)
    # 6. Qualitative results
    qualitative_examples(model, val_loader, device, base, threshold=0.34, count=10)
    # 7. Training curves
    training_curves(base)
    # 8. Ablation study
    ablation_study(model, val_loader, device, base, final_thr=0.34)
    print("Paper outputs saved to:", base)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    with torch.no_grad():
        main()
