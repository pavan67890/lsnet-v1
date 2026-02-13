import os
import random
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import subprocess
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
import numpy as np
from models.lsnet import LSNet
from utils.utils import set_seed, WarmupCosine, count_params
import sysu_config as config
from contextlib import nullcontext
from datasets.cd_dataset import PatchChangeDetectionDataset
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    targets = targets.float()
    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    dice = (2.0 * intersection + eps) / union
    return 1.0 - dice.mean()
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
def lovasz_hinge_masked(logits, labels, mask, per_image=True):
    if per_image:
        losses = []
        for logit, lab, mk in zip(logits, labels, mask):
            l_flat = logit.view(-1)
            y_flat = lab.view(-1)
            m_flat = mk.view(-1).bool()
            if m_flat.sum() == 0:
                # fallback to full image
                losses.append(lovasz_hinge_flat(l_flat, y_flat))
            else:
                losses.append(lovasz_hinge_flat(l_flat[m_flat], y_flat[m_flat]))
        if len(losses) == 0:
            return logits.new_tensor(0.0)
        return torch.mean(torch.stack(losses))
    else:
        l_flat = logits.view(-1)
        y_flat = labels.view(-1)
        m_flat = mask.view(-1).bool()
        if m_flat.sum() == 0:
            return lovasz_hinge_flat(l_flat, y_flat)
        return lovasz_hinge_flat(l_flat[m_flat], y_flat[m_flat])

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
    # compute class-balanced sampler: 50% change vs 50% no-change
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
    persistent_workers_cfg = getattr(config, "PERSISTENT_WORKERS", False)
    prefetch_factor_cfg = getattr(config, "PREFETCH_FACTOR", None)
    pin_memory_cfg = getattr(config, "PIN_MEMORY", True)
    loader_args = dict(
        batch_size=getattr(config, "BATCH_SIZE", 8),
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
        batch_size=getattr(config, "BATCH_SIZE", 8),
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

def compute_pos_weight(patches_root):
    train_root = os.path.join(patches_root, "train")
    mdir = os.path.join(train_root, "mask")
    files = [f for f in os.listdir(mdir) if f.lower().endswith(".npy")]
    pos = 0.0
    neg = 0.0
    for f in files:
        m = np.load(os.path.join(mdir, f)).astype(np.float32)
        p = (m > 0.5).sum()
        n = (m <= 0.5).sum()
        pos += float(p)
        neg += float(n)
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / max(1.0, pos))

def main():
    set_seed(getattr(config, "SEED", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    pos_weight = torch.tensor(3.0, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def loss_fn(logits, targets):
        return 0.5 * bce(logits, targets) + 0.5 * lovasz_hinge(logits, (targets > 0.5).float())
    optimizer = optim.AdamW(model.parameters(), lr=getattr(config, "LR", 1e-4), weight_decay=getattr(config, "WEIGHT_DECAY", 1e-2))
    scaler = GradScaler()
    swa_model = None
    warmup_epochs = getattr(config, "WARMUP_EPOCHS", 5)
    phase1_epochs = getattr(config, "EPOCHS", 150)
    swa_start = getattr(config, "SWA_START", 60)
    swa_lr = getattr(config, "SWA_LR", 1e-4)
    scheduler = WarmupCosine(optimizer, warmup_epochs=warmup_epochs, total_epochs=phase1_epochs, min_lr=1e-6)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_path = os.path.join(config.LOG_DIR, "sysu_training_log.csv")
    diag_path = os.path.join(config.LOG_DIR, "sysu_diag.txt")
    print("SYSU clean run: geometric aug only, histogram matching in training")
    try:
        with open(diag_path, "w") as _f:
            _f.write("")
    except Exception:
        pass
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou,best_threshold,epoch_time\n")
    best_iou = -1.0
    best_thr = 0.9
    start_epoch = 0
    clean_run = os.environ.get("SYSU_CLEAN_RUN", "0") == "1"
    # Resume from last checkpoint if available, else fallback to best
    last_ckpt = os.path.join(config.CHECKPOINT_DIR, "sysu_last.pth")
    resume_ok = False
    if (not clean_run) and os.path.exists(last_ckpt):
        try:
            payload = torch.load(last_ckpt, map_location=device)
            state = payload.get("model", None)
            if state is not None:
                model.load_state_dict(state)
            if "optimizer" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer"])
                except Exception:
                    pass
            bi = payload.get("best_iou", payload.get("iou", -1.0))
            if bi is not None:
                best_iou = float(bi)
            bt = payload.get("best_thr", best_thr)
            if bt is not None:
                best_thr = float(bt)
            start_epoch = int(payload.get("epoch", 0))
            print(f"Resumed from epoch: {start_epoch}")
            print(f"Current best IoU: {best_iou}")
            print(f"Current best threshold: {best_thr}")
            resume_ok = True
        except Exception:
            resume_ok = False
    # resume logic handled above; clean run skips resume
    train_loader, val_loader = build_loaders(config.PATCH_ROOT, max_val_patches=None)
    print("CUDA:", torch.cuda.is_available())
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)
    print("AMP:", "Enabled")
    print("num_workers:", train_loader.num_workers)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Batch size:", train_loader.batch_size)
    print("Train batches:", len(train_loader))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
        util_lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        print("GPU utilization:", ", ".join(util_lines))
        if any(int(u) < 50 for u in util_lines):
            print("GPU utilization below 50%")
        try:
            with open(diag_path, "a") as _f:
                _f.write("GPU utilization: " + ", ".join(util_lines) + "\n")
        except Exception:
            pass
    except Exception:
        print("GPU utilization check failed")
    eff_bs = train_loader.batch_size * int(getattr(config, "GRAD_ACCUM_STEPS", 1))
    print("Effective batch size:", eff_bs)
    last_losses = []
    inc_streak = 0
    prev_loss = None
    one_epoch = os.environ.get("SYSU_ONE_EPOCH", "0") == "1"
    phase1_limit = os.environ.get("SYSU_PHASE1_EPOCHS", "")
    if one_epoch:
        end_epoch = start_epoch + 1
    elif phase1_limit.isdigit():
        end_epoch = min(phase1_epochs, start_epoch + int(phase1_limit))
    else:
        end_epoch = phase1_epochs
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        ga = max(1, getattr(config, "GRAD_ACCUM_STEPS", 1))
        pbar = tqdm(range(len(train_loader)), total=len(train_loader), dynamic_ncols=True)
        it = iter(train_loader)
        for i in pbar:
            t_data_s = time.perf_counter()
            t1, t2, m = next(it)
            t_data_e = time.perf_counter()
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            ctx = autocast() if device.type=='cuda' else nullcontext()
            t_comp_s = time.perf_counter()
            with ctx:
                logits, _ = model(t1, t2)
                loss = loss_fn(logits, m) / ga
            t_comp_e = time.perf_counter()
            if i < 20:
                print(f"Batch {i+1} data_time {t_data_e - t_data_s:.3f}s compute_time {t_comp_e - t_comp_s:.3f}s")
                try:
                    with open(diag_path, "a") as _f:
                        _f.write(f"Batch {i+1} data_time {t_data_e - t_data_s:.6f}s compute_time {t_comp_e - t_comp_s:.6f}s\n")
                except Exception:
                    pass
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
            nf_path = os.path.join(config.CHECKPOINT_DIR, "sysu_last.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "best_thr": best_thr,
            }, nf_path)
            print("Non-finite loss detected. Saving and stopping.")
            break
        last_losses.append(avg_loss)
        if prev_loss is not None:
            if avg_loss > prev_loss:
                inc_streak += 1
            else:
                inc_streak = 0
        prev_loss = avg_loss
        if inc_streak >= 10:
            is_path = os.path.join(config.CHECKPOINT_DIR, "sysu_last.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "best_thr": best_thr,
            }, is_path)
            print("Loss increased for 10 consecutive epochs. Saving and stopping.")
            break
        if epoch < swa_start:
            scheduler.step()
        if epoch == swa_start:
            swa_model = AveragedModel(model)
            for g in optimizer.param_groups:
                g['lr'] = swa_lr
        if swa_model is not None:
            swa_model.update_parameters(model)
        _, vloader = build_loaders(config.PATCH_ROOT, max_val_patches=None)
        p = r = f1 = iou = None
        do_val = True
        if do_val:
            model.eval()
            thresholds = torch.arange(0.10, 0.91, 0.02, device=device)
            tp = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
            fp = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
            fn = torch.zeros_like(thresholds, dtype=torch.float64, device=device)
            val_pbar = tqdm(vloader, total=len(vloader), dynamic_ncols=True, desc=f"Val Epoch {epoch+1}", leave=False, mininterval=0.5)
            with torch.no_grad():
                for t1, t2, m in val_pbar:
                    t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
                    logits, probs = model(t1, t2)
                    pos = (m > 0.5)
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
                best_path = os.path.join(config.CHECKPOINT_DIR, "sysu_best.pth")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1, "iou": iou, "f1": f1, "best_thr": best_thr},
                           best_path)
            print(f"Epoch {epoch+1} Validation:")
            print(f"Precision: {p:.4f}")
            print(f"Recall:    {r:.4f}")
            print(f"F1:        {f1:.4f}")
            print(f"IoU:       {iou:.4f}")
            print(f"Best threshold: {best_thr:.2f}")
        epoch_time = time.time() - epoch_start
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{'' if p is None else p},{'' if r is None else r},{'' if f1 is None else f1},{'' if iou is None else iou},{best_thr},{epoch_time}\n")
        print(f"Epoch {epoch+1} completed | train_loss {avg_loss:.4f} | epoch_time {epoch_time:.2f}")
        if p is not None:
            print(f"Epoch {epoch+1} | F1 {f1:.4f} | IoU {iou:.4f}")
        save_obj = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iou": iou,
            "f1": f1,
            "best_thr": best_thr,
        }
        last_path = os.path.join(config.CHECKPOINT_DIR, "sysu_last.pth")
        torch.save(save_obj, last_path)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if swa_model is not None:
        update_bn(train_loader, swa_model)
        swa_path = os.path.join(config.CHECKPOINT_DIR, "sysu_swa_final.pth")
        torch.save({"model": swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(), "epoch": phase1_epochs},
                   swa_path)
    print("SYSU Training complete.")

if __name__ == "__main__":
    main()
