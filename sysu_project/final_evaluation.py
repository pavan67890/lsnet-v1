import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.cd_dataset import PatchChangeDetectionDataset
from models.lsnet import LSNet
import sysu_config as config

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = r"S:\sota\sysu_project\checkpoints\sysu_best.pth"
REPORT_DIR = r"S:\sota\sysu_project\final_report"
EX_DIR = os.path.join(REPORT_DIR, "examples")

def ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EX_DIR, exist_ok=True)

def load_model():
    model = LSNet().to(DEVICE)
    payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model

def build_loader(root, batch_size=2, shuffle=False):
    ds = PatchChangeDetectionDataset(root, mode="test")
    num_workers = getattr(config, "NUM_WORKERS", 2)
    pin_memory = getattr(config, "PIN_MEMORY", True)
    persistent = getattr(config, "PERSISTENT_WORKERS", True)
    prefetch = getattr(config, "PREFETCH_FACTOR", 2)
    args = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0 and persistent:
        args["persistent_workers"] = True
    if num_workers > 0 and prefetch is not None:
        args["prefetch_factor"] = int(prefetch)
    return ds, DataLoader(ds, **args)

def get_logits_probs(model, t1, t2):
    logits, probs = model(t1, t2)
    return logits, probs

def metrics_from_counts(tp, tn, fp, fn):
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou, accuracy=accuracy, specificity=specificity)

def compute_global_confusion(model, loader, threshold=0.5):
    TP = TN = FP = FN = 0
    with torch.no_grad():
        for t1, t2, m in tqdm(loader, desc="Global eval", dynamic_ncols=True):
            t1 = t1.to(DEVICE, non_blocking=True)
            t2 = t2.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)
            logits, probs = get_logits_probs(model, t1, t2)
            pred = (probs > threshold).float()
            pos = (m > 0.5).float()
            neg = 1.0 - pos
            TP += (pred * pos).sum().item()
            TN += ((1.0 - pred) * neg).sum().item()
            FP += (pred * neg).sum().item()
            FN += ((1.0 - pred) * pos).sum().item()
    return TP, TN, FP, FN

def plot_confusion(tp, tn, fp, fn, out_path):
    # normalized confusion matrix for binary class
    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=np.float64)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_xticklabels(['Pred 0','Pred 1'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['True 0','True 1'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def sample_subset(ds, k=200):
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)
    idxs = idxs[:min(k, n)]
    sub = Subset(ds, idxs)
    return sub

def collect_scores_labels(model, loader):
    scores = []
    labels = []
    with torch.no_grad():
        for t1, t2, m in tqdm(loader, desc="Collect ROC/PR", dynamic_ncols=True):
            t1 = t1.to(DEVICE, non_blocking=True)
            t2 = t2.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)
            logits, probs = get_logits_probs(model, t1, t2)
            scores.append(probs.detach().flatten().cpu())
            labels.append((m > 0.5).float().flatten().cpu())
    scores = torch.cat(scores, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return scores, labels

def plot_roc(scores, labels, out_path):
    # compute ROC by sweeping thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    tpr = []
    fpr = []
    pos = (labels > 0.5).astype(np.uint8)
    neg = 1 - pos
    P = pos.sum()
    N = neg.sum()
    for thr in thresholds:
        pred = (scores > thr).astype(np.uint8)
        tp = (pred & pos).sum()
        fp = (pred & neg).sum()
        fn = ((1 - pred) & pos).sum()
        tn = ((1 - pred) & neg).sum()
        tpr.append(tp / max(1, P))
        fpr.append(fp / max(1, N))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_pr(scores, labels, out_path):
    thresholds = np.linspace(0.0, 1.0, 101)
    precision = []
    recall = []
    pos = (labels > 0.5).astype(np.uint8)
    for thr in thresholds:
        pred = (scores > thr).astype(np.uint8)
        tp = (pred & pos).sum()
        fp = (pred & (1 - pos)).sum()
        fn = ((1 - pred) & pos).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        precision.append(p)
        recall.append(r)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(recall, precision, label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def get_tta_prediction(model, img1, img2):
    preds = []
    transforms = [
        (0, None), (1, None), (2, None), (3, None),
        (0, 2), (0, 3), (1, 2), (1, 3)
    ]
    with torch.no_grad():
        for k, flip in transforms:
            x1 = img1.clone()
            x2 = img2.clone()
            if flip:
                x1 = torch.flip(x1, (flip,))
                x2 = torch.flip(x2, (flip,))
            if k > 0:
                x1 = torch.rot90(x1, k, (2, 3))
                x2 = torch.rot90(x2, k, (2, 3))
            logits, probs = get_logits_probs(model, x1, x2)
            p = probs
            if k > 0:
                p = torch.rot90(p, -k, (2, 3))
            if flip:
                p = torch.flip(p, (flip,))
            preds.append(p)
    return torch.stack(preds).mean(dim=0)

def ablation_base_vs_tta(model, ds):
    # Base metrics (single pass) on full set
    _, full_loader = build_loader(ds.root, batch_size=2, shuffle=False)
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for t1, t2, m in tqdm(full_loader, desc="Ablation Base", dynamic_ncols=True):
            t1 = t1.to(DEVICE, non_blocking=True)
            t2 = t2.to(DEVICE, non_blocking=True)
            m = m.to(DEVICE, non_blocking=True)
            logits, probs = get_logits_probs(model, t1, t2)
            pred = (probs > 0.5).float()
            pos = (m > 0.5).float()
            neg = 1.0 - pos
            tp += (pred * pos).sum().item()
            tn += ((1.0 - pred) * neg).sum().item()
            fp += (pred * neg).sum().item()
            fn += ((1.0 - pred) * pos).sum().item()
    base = metrics_from_counts(tp, tn, fp, fn)
    # TTA metrics on full set (may be slower but thorough)
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        for idx in tqdm(range(len(ds)), desc="Ablation TTA", dynamic_ncols=True):
            t1, t2, m = ds[idx]
            t1 = t1.unsqueeze(0).to(DEVICE, non_blocking=True)
            t2 = t2.unsqueeze(0).to(DEVICE, non_blocking=True)
            m = m.unsqueeze(0).to(DEVICE, non_blocking=True)
            p = get_tta_prediction(model, t1, t2)
            pred = (p > 0.5).float()
            pos = (m > 0.5).float()
            neg = 1.0 - pos
            tp += (pred * pos).sum().item()
            tn += ((1.0 - pred) * neg).sum().item()
            fp += (pred * neg).sum().item()
            fn += ((1.0 - pred) * pos).sum().item()
    tta = metrics_from_counts(tp, tn, fp, fn)
    return base, tta

def save_examples(model, ds, out_dir, count=5):
    os.makedirs(out_dir, exist_ok=True)
    pick = list(range(min(count, len(ds))))
    with torch.no_grad():
        for i in pick:
            t1, t2, m = ds[i]
            t1b = t1.unsqueeze(0).to(DEVICE, non_blocking=True)
            t2b = t2.unsqueeze(0).to(DEVICE, non_blocking=True)
            logits, probs = get_logits_probs(model, t1b, t2b)
            pred = (probs > 0.5).float()
            t1_np = t1.cpu().numpy().transpose(1,2,0)
            t2_np = t2.cpu().numpy().transpose(1,2,0)
            gt_np = (m.cpu().numpy()[0] > 0.5).astype(np.uint8) * 255
            pr_np = (pred.cpu().numpy()[0,0] > 0.5).astype(np.uint8) * 255
            fig, axes = plt.subplots(1,4, figsize=(10,3))
            axes[0].imshow(np.clip(t1_np, 0, 1)); axes[0].set_title("T1"); axes[0].axis('off')
            axes[1].imshow(np.clip(t2_np, 0, 1)); axes[1].set_title("T2"); axes[1].axis('off')
            axes[2].imshow(gt_np, cmap='gray'); axes[2].set_title("GT"); axes[2].axis('off')
            axes[3].imshow(pr_np, cmap='gray'); axes[3].set_title("Pred"); axes[3].axis('off')
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"example_{i+1}.png"))
            plt.close(fig)

def main():
    try:
        ensure_dirs()
        model = load_model()
        test_root = os.path.join(config.PATCH_ROOT, "val")
        ds, loader = build_loader(test_root, batch_size=2, shuffle=False)
        tp, tn, fp, fn = compute_global_confusion(model, loader, threshold=0.5)
        metrics = metrics_from_counts(tp, tn, fp, fn)
        cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
        plot_confusion(tp, tn, fp, fn, cm_path)
        print("Global Metrics:", {k: f"{v:.4f}" for k,v in metrics.items()})
        print("Saved:", cm_path)
        with open(os.path.join(REPORT_DIR, "results.txt"), "w") as rf:
            rf.write("Global Metrics\n")
            for k, v in metrics.items():
                rf.write(f"{k}: {v:.6f}\n")
        subset = sample_subset(ds, k=200)
        sub_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=getattr(config, "NUM_WORKERS", 2), pin_memory=getattr(config, "PIN_MEMORY", True))
        scores, labels = collect_scores_labels(model, sub_loader)
        roc_path = os.path.join(REPORT_DIR, "roc_curve.png")
        pr_path = os.path.join(REPORT_DIR, "pr_curve.png")
        plot_roc(scores, labels, roc_path)
        plot_pr(scores, labels, pr_path)
        print("Saved:", roc_path)
        print("Saved:", pr_path)
        base, tta = ablation_base_vs_tta(model, ds)
        print("Ablation Study (Base vs TTA)")
        print(f"{'Metric':<12} {'Base':>10} {'TTA':>10}")
        for k in ["precision","recall","f1","iou","accuracy","specificity"]:
            print(f"{k:<12} {base[k]:>10.4f} {tta[k]:>10.4f}")
        with open(os.path.join(REPORT_DIR, "results.txt"), "a") as rf:
            rf.write("\nAblation Study (Base vs TTA)\n")
            for k in ["precision","recall","f1","iou","accuracy","specificity"]:
                rf.write(f"{k}: base={base[k]:.6f}, tta={tta[k]:.6f}\n")
        save_examples(model, ds, EX_DIR, count=5)
        print("Examples saved to:", EX_DIR)
        print("Final evaluation complete. Reports saved to:", REPORT_DIR)
    except Exception as e:
        import traceback
        print("Final evaluation error:", str(e))
        traceback.print_exc()

if __name__ == "__main__":
    main()
