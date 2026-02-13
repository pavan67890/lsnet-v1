import os
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
import numpy as np
from models.lsnet import LSNet
from utils.utils import set_seed, WarmupCosine
import sysu_config as config
from contextlib import nullcontext
from datasets.cd_dataset import PatchChangeDetectionDataset
import torch.nn.functional as F

def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def lovasz_grad(gt_sorted):
    gts = gt_sorted
    p = gts.size(0)
    gts_sum = gts.sum()
    intersection = gts_sum - gts.float().cumsum(0)
    union = gts_sum + (1 - gts).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:p-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits.float() * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        losses = []
        for logit, lab in zip(logits, labels):
            ls, lb = flatten_binary_scores(logit, lab, ignore)
            if lb.numel() == 0:
                continue
            losses.append(lovasz_hinge_flat(ls, lb))
        if len(losses) == 0:
            return logits.new_tensor(0.0)
        return torch.mean(torch.stack(losses))
    else:
        ls, lb = flatten_binary_scores(logits, labels, ignore)
        if lb.numel() == 0:
            return logits.new_tensor(0.0)
        return lovasz_hinge_flat(ls, lb)

def build_loaders(patches_root, max_val_patches=None):
    train_root = os.path.join(patches_root, "train")
    val_root = os.path.join(patches_root, "val")
    if not os.path.isdir(os.path.join(val_root, "mask")):
        val_root = train_root
    train_ds = PatchChangeDetectionDataset(train_root, mode="train")
    val_ds = PatchChangeDetectionDataset(val_root, mode="test")
    pos_counts = 0
    neg_counts = 0
    weights = []
    for fname in train_ds.index_map:
        m = torch.from_numpy(__import__("numpy").load(os.path.join(train_ds.mask_dir, fname)).astype(__import__("numpy").float32))
        pos = (m > 0.5).float().mean().item()
        is_change = pos >= 0.01
        if is_change:
            pos_counts += 1
        else:
            neg_counts += 1
    change_w = 1.0
    nochange_w = (pos_counts / max(1, neg_counts)) if neg_counts > 0 else 1.0
    for fname in train_ds.index_map:
        m = torch.from_numpy(__import__("numpy").load(os.path.join(train_ds.mask_dir, fname)).astype(__import__("numpy").float32))
        pos = (m > 0.5).float().mean().item()
        is_change = pos >= 0.01
        weights.append(change_w if is_change else nochange_w)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    num_workers = getattr(config, "NUM_WORKERS", 2)
    persistent_workers_cfg = getattr(config, "PERSISTENT_WORKERS", True)
    prefetch_factor_cfg = getattr(config, "PREFETCH_FACTOR", 2)
    pin_memory_cfg = getattr(config, "PIN_MEMORY", True)
    loader_args = dict(
        batch_size=getattr(config, "BATCH_SIZE", 4),
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_cfg,
    )
    if num_workers > 0 and persistent_workers_cfg:
        loader_args["persistent_workers"] = True
    if num_workers > 0 and prefetch_factor_cfg is not None:
        loader_args["prefetch_factor"] = int(prefetch_factor_cfg)
    train_loader = DataLoader(train_ds, **loader_args)
    if max_val_patches is not None:
        n = len(val_ds)
        if n > max_val_patches:
            idx = random.sample(range(n), max_val_patches)
            val_ds = Subset(val_ds, idx)
    val_args = dict(
        batch_size=getattr(config, "BATCH_SIZE", 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory_cfg,
    )
    if num_workers > 0 and persistent_workers_cfg:
        val_args["persistent_workers"] = True
    if num_workers > 0 and prefetch_factor_cfg is not None:
        val_args["prefetch_factor"] = int(prefetch_factor_cfg)
    val_loader = DataLoader(val_ds, **val_args)
    return train_loader, val_loader

def get_edge_mask(mask):
    k = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]], device=mask.device)
    k = k.view(1, 1, 3, 3)
    e = F.conv2d(mask, k, padding=1)
    e = e.abs()
    mx = e.amax(dim=(2,3), keepdim=True) + 1e-6
    e = e / mx
    e = e.clamp(0.0, 1.0)
    return e

def main():
    set_seed(getattr(config, "SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "sysu_best.pth")
    payload = torch.load(ckpt_path, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    def criterion(logits, targets):
        return 0.5 * bce(logits, targets) + 0.5 * lovasz_hinge(logits, (targets > 0.5).float())
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=getattr(config, "WEIGHT_DECAY", 1e-2))
    scaler = GradScaler()
    swa_model = None
    total_epochs = 15
    scheduler = WarmupCosine(optimizer, warmup_epochs=0, total_epochs=total_epochs, min_lr=1e-6)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "sysu_training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou,best_threshold,epoch_time\n")
    best_iou = -1.0
    best_thr = 0.5
    start_epoch = 0
    train_loader, val_loader = build_loaders(config.PATCH_ROOT, max_val_patches=None)
    print("CUDA:", torch.cuda.is_available())
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
    print("AMP:", "Enabled")
    print("num_workers:", train_loader.num_workers)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Batch size:", train_loader.batch_size)
    print("Train batches:", len(train_loader))
    eff_bs = train_loader.batch_size * int(getattr(config, "GRAD_ACCUM_STEPS", 1))
    print("Effective batch size:", eff_bs)
    for epoch in range(start_epoch, start_epoch + total_epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        ga = max(1, getattr(config, "GRAD_ACCUM_STEPS", 1))
        pbar = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True)
        for i, (t1, t2, m) in enumerate(pbar):
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            edge_mask = get_edge_mask(m)
            ctx = autocast() if device.type=='cuda' else nullcontext()
            with ctx:
                logits, _ = model(t1, t2)
                base = criterion(logits, m)
                edge = criterion(logits * edge_mask, m * edge_mask)
                loss = (base + 1.0 * edge) / ga
            if i % 100 == 0:
                try:
                    pbar.set_description(f"Epoch {epoch+1} | Loss {loss.item():.4f}")
                except Exception:
                    pass
            scaler.scale(loss).backward()
            if (i + 1) % ga == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * ga * t1.size(0)
        try:
            pbar.close()
        except Exception:
            pass
        avg_loss = total_loss / len(train_loader.dataset)
        if not torch.isfinite(torch.tensor(avg_loss)):
            print("Non-finite loss detected. Stopping.")
            break
        if epoch < total_epochs:
            scheduler.step()
        _, vloader = build_loaders(config.PATCH_ROOT, max_val_patches=None)
        p = r = f1 = iou = None
        model.eval()
        thresholds = torch.arange(0.10, 0.91, 0.02, device=device)
        tp = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
        fp = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
        fn = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
        val_pbar = tqdm(vloader, total=len(vloader), dynamic_ncols=True, desc=f"Val Epoch {epoch+1}", leave=False, mininterval=0.5)
        with torch.no_grad():
            for t1, t2, mm in val_pbar:
                t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); mm = mm.to(device, non_blocking=True)
                logits, probs = model(t1, t2)
                pos = (mm > 0.5)
                neg = ~pos
                for i in range(thresholds.numel()):
                    pred = probs > thresholds[i]
                    tp[i] += (pred & pos).sum()
                    fp[i] += (pred & neg).sum()
                    fn[i] += ((~pred) & pos).sum()
        eps = 1e-8
        precision_arr = tp / (tp + fp + eps)
        recall_arr = tp / (tp + fn + eps)
        f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + eps)
        iou_arr = tp / (tp + fp + fn + eps)
        best_idx = torch.argmax(iou_arr).item()
        best_thr = thresholds[best_idx].item()
        p = precision_arr[best_idx].item()
        r = recall_arr[best_idx].item()
        f1 = f1_arr[best_idx].item()
        iou = iou_arr[best_idx].item()
        if iou > best_iou:
            best_iou = iou
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1, "iou": iou, "f1": f1, "best_thr": best_thr},
                       os.path.join(config.CHECKPOINT_DIR, "sysu_sharp_best.pth"))
        epoch_time = time.time() - epoch_start
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{'' if p is None else p},{'' if r is None else r},{'' if f1 is None else f1},{'' if iou is None else iou},{best_thr},{epoch_time}\n")
        print(f"Epoch {epoch+1} completed | train_loss {avg_loss:.4f} | epoch_time {epoch_time:.2f}")
        if p is not None:
            print(f"Epoch {epoch+1} | F1 {f1:.4f} | IoU {iou:.4f}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print("Sharpen Training complete.")

if __name__ == "__main__":
    main()
