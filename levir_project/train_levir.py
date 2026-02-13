import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
import numpy as np
from models.lsnet import LSNet
from losses.focal_tversky import FocalTverskyLoss
from utils.utils import set_seed, metrics, WarmupCosine, count_params
import config
from contextlib import nullcontext

def build_loaders(max_val_patches=3000):
    from datasets.cd_dataset import PatchChangeDetectionDataset
    train_root = os.path.join(config.PATCHES_ROOT, "train")
    val_root = os.path.join(config.PATCHES_ROOT, "val")
    if not os.path.isdir(os.path.join(val_root, "mask")):
        val_root = train_root
    train_ds = PatchChangeDetectionDataset(train_root, mode="train")
    val_ds = PatchChangeDetectionDataset(val_root, mode="test")
    # Weighted sampler by positive ratio
    weights = []
    for fname in train_ds.index_map:
        m = torch.from_numpy(__import__("numpy").load(os.path.join(train_ds.mask_dir, fname)).astype(__import__("numpy").float32))
        pos = (m > 0.5).float().mean().item()
        w = 10.0 if pos > 0.0 else 1.0
        weights.append(w)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    num_workers = config.num_workers
    loader_args = dict(
        batch_size=config.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_args["persistent_workers"] = True
        loader_args["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **loader_args)
    if max_val_patches is not None:
        n = len(val_ds)
        if n > max_val_patches:
            idx = random.sample(range(n), max_val_patches)
            val_ds = Subset(val_ds, idx)
    val_args = dict(
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        val_args["persistent_workers"] = True
        val_args["prefetch_factor"] = 2
    val_loader = DataLoader(val_ds, **val_args)
    return train_loader, val_loader

def train():
    set_seed(config.seed)
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    pos_weight = torch.tensor(8.0, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def loss_fn(logits, targets):
        return 0.7 * ft(logits, targets) + 0.3 * bce(logits, targets)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=config.amp, device='cuda' if device.type=='cuda' else 'cpu')
    swa_model = None
    warmup_epochs = 5
    phase1_epochs = 75
    swa_start = 60
    swa_lr = 1e-4
    scheduler = WarmupCosine(optimizer, warmup_epochs=warmup_epochs, total_epochs=phase1_epochs, min_lr=1e-6)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, "levir_training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou,best_threshold,epoch_time\n")
    best_iou = -1.0
    best_thr = 0.5
    # automatic resume
    start_epoch = 0
    last_ckpt = os.path.join(config.CHECKPOINTS_DIR, "levir_last.pth")
    if os.path.exists(last_ckpt):
        try:
            payload = torch.load(last_ckpt, map_location=device)
            state = payload.get("model", None)
            if state is not None:
                model.load_state_dict(state)
            if "optimizer" in payload:
                optimizer.load_state_dict(payload["optimizer"])
            if "scheduler" in payload:
                try:
                    scheduler.load_state_dict(payload["scheduler"])
                except Exception:
                    pass
            if "scaler" in payload:
                try:
                    scaler.load_state_dict(payload["scaler"])
                except Exception:
                    pass
            start_epoch = int(payload.get("epoch", 0))
            print("Resuming from epoch", start_epoch + 1)
        except Exception:
            pass
    train_loader, val_loader = build_loaders(max_val_patches=None)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Batch size:", train_loader.batch_size)
    print("num_workers:", train_loader.num_workers)
    print("Train batches:", len(train_loader))
    eff_bs = train_loader.batch_size * int(getattr(config, "grad_accum_steps", 1))
    print("Effective batch size:", eff_bs)
    if torch.cuda.is_available():
        try:
            print("GPU:", torch.cuda.get_device_name(0))
        except Exception:
            pass
    last_losses = []
    inc_streak = 0
    prev_loss = None
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        ga = max(1, config.grad_accum_steps)
        pbar = tqdm(train_loader, total=len(train_loader), dynamic_ncols=True)
        for i, (t1, t2, m) in enumerate(pbar):
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            with autocast('cuda') if device.type=='cuda' else torch.autocast(device_type='cpu', dtype=torch.float32):
                logits, _ = model(t1, t2)
                loss = loss_fn(logits, m) / ga
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
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "best_thr": best_thr,
            }, os.path.join(config.CHECKPOINTS_DIR, "levir_last.pth"))
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
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_iou": best_iou,
                "best_thr": best_thr,
            }, os.path.join(config.CHECKPOINTS_DIR, "levir_last.pth"))
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
        _, vloader = build_loaders(max_val_patches=None)
        p = r = f1 = iou = None
        do_val = False
        if (epoch + 1) < 60:
            if (epoch + 1) % 5 == 0:
                do_val = True
        else:
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
                    ctx = autocast('cuda') if device.type=='cuda' else nullcontext()
                    with ctx:
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
            best_idx = torch.argmax(f1_arr).item()
            best_thr = thresholds[best_idx].item()
            p = precision_arr[best_idx].item()
            r = recall_arr[best_idx].item()
            f1 = f1_arr[best_idx].item()
            iou = iou_arr[best_idx].item()
            if iou > best_iou:
                best_iou = iou
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1, "iou": iou, "f1": f1, "best_thr": best_thr},
                           os.path.join(config.CHECKPOINTS_DIR, "levir_best.pth"))
            print(f"Epoch {epoch+1} Validation:")
            print(f"Precision: {p:.4f}")
            print(f"Recall:    {r:.4f}")
            print(f"F1:        {f1:.4f}")
            print(f"IoU:       {iou:.4f}")
            print(f"Best threshold: {best_thr:.2f}")
        # epoch time and logging
        epoch_time = time.time() - epoch_start
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss},{'' if p is None else p},{'' if r is None else r},{'' if f1 is None else f1},{'' if iou is None else iou},{best_thr},{epoch_time}\n")
        print(f"Epoch {epoch+1} completed | train_loss {avg_loss:.4f} | epoch_time {epoch_time:.2f}")
        if p is not None and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} | F1 {f1:.4f} | IoU {iou:.4f}")
        # save last checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_iou": best_iou,
            "best_thr": best_thr,
        }, os.path.join(config.CHECKPOINTS_DIR, "levir_last.pth"))
    # finalize SWA
    if swa_model is not None:
        update_bn(train_loader, swa_model)
        torch.save({"model": swa_model.module.state_dict() if hasattr(swa_model, 'module') else swa_model.state_dict(), "epoch": config.epochs},
                   os.path.join(config.CHECKPOINTS_DIR, "levir_swa_final.pth"))
    print("Training complete.")

if __name__ == "__main__":
    train()
