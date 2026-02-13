import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.lsnet import LSNet
from losses.lovasz import lovasz_hinge
from losses.focal_tversky import FocalTverskyLoss
from datasets.cd_dataset import ChangeDetectionDataset
from utils import set_seed, count_params, metrics, WarmupCosine
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
import random
import subprocess

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataloaders(max_val_patches=3000, batch_size=None, patches_root=None):
    default_root = os.path.join(os.path.dirname(config.data_root), "whu_patches")
    root = patches_root if patches_root else default_root
    use_patches = os.path.isdir(os.path.join(root, "t1")) and os.path.isdir(os.path.join(root, "t2")) and os.path.isdir(os.path.join(root, "mask"))
    if use_patches:
        from datasets.cd_dataset import PatchChangeDetectionDataset
        train_ds = PatchChangeDetectionDataset(root, mode="train")
        test_ds = PatchChangeDetectionDataset(root, mode="test")
    else:
        train_ds = ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="train", patch_size=config.patch_size, stride=config.train_stride)
        test_ds = ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="test", patch_size=config.patch_size, stride=config.test_stride)
    if max_val_patches is not None:
        n = len(test_ds)
        if n > max_val_patches:
            from torch.utils.data import Subset
            idx = random.sample(range(n), max_val_patches)
            test_ds = Subset(test_ds, idx)
    workers = 2
    bs = config.micro_batch_size if batch_size is None else batch_size
    train_kwargs = dict(batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_kwargs = dict(batch_size=config.micro_batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, **train_kwargs)
    test_loader = DataLoader(test_ds, **test_kwargs)
    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    ga = max(1, config.grad_accum_steps)
    desc = f"Train Epoch {epoch + 1}" if epoch is not None else "Train"
    for i, (t1, t2, m) in enumerate(tqdm(loader, desc=desc, leave=False)):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            logits, _ = model(t1, t2)
            loss = criterion(logits, m) / ga
        if torch.isnan(loss).any():
            print("NaN loss detected — stopping epoch", flush=True)
            return 0
        scaler.scale(loss).backward()
        if (i + 1) % ga == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * ga * t1.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    p_sum = r_sum = f1_sum = iou_sum = 0.0
    n = 0
    with torch.no_grad():
        for t1, t2, m in tqdm(loader, desc="Validation", leave=False):
            t1 = t1.to(device, non_blocking=True)
            t2 = t2.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                logits, probs = model(t1, t2)
            p, r, f1, iou = metrics(probs, m)
            p_sum += p
            r_sum += r
            f1_sum += f1
            iou_sum += iou
            n += 1
    return p_sum / n, r_sum / n, f1_sum / n, iou_sum / n

def main():
    set_seed(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    device = get_device()
    model = LSNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    ft = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    def composite_loss(logits, targets):
        return 0.4 * bce(logits, targets) + 0.4 * lovasz_hinge(logits, targets) + 0.2 * ft(logits, targets)
    criterion = composite_loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=25)
    scaler = GradScaler(device='cuda')
    start_epoch = 0
    best_iou = -1.0
    os.makedirs("checkpoints", exist_ok=True)
    phase2_resume = r"S:\sota\project\checkpoints\epoch 45 iou 0.857.pth"
    if os.path.exists(phase2_resume):
        try:
            payload = torch.load(phase2_resume, map_location=device)
            state = payload.get("model", None)
            if state is None:
                print("Checkpoint missing model state_dict. Abort.", flush=True)
                return
            model.load_state_dict(state)
            if "optimizer" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer"])
                except Exception:
                    pass
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
            start_epoch = int(payload.get("epoch", 45))
            bi = payload.get("iou", None)
            if bi is not None:
                best_iou = float(bi)
            print(f"Resuming training from epoch {start_epoch + 1}", flush=True)
        except Exception as e:
            print("Failed to load checkpoint:", str(e), flush=True)
            return
    else:
        print("Checkpoint not found. Abort.", flush=True)
        return
    config.grad_accum_steps = 2
    config.grad_accum_steps = 2
    chosen_bs = 4
    phase1_root = os.path.join(os.path.dirname(config.data_root), "whu_patches_s256")
    phase2_root = os.path.join(os.path.dirname(config.data_root), "whu_patches")
    def ensure_patches(stride, dest_root):
        t1_dir = os.path.join(dest_root, "t1")
        ok = os.path.isdir(t1_dir) and len(os.listdir(t1_dir)) > 0
        if ok:
            return
        cmd = [
            "s:\\sota\\ffbdnetx_env\\Scripts\\python.exe",
            "-u",
            "s:\\sota\\project\\prepare_patches.py",
            "--stride", str(stride),
            "--patch_size", str(config.patch_size),
            "--dest", dest_root
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            pass
    if start_epoch < 20:
        ensure_patches(256, phase1_root)
        current_root = phase1_root
    else:
        current_root = phase2_root
    val_cap = 3000 if start_epoch < (config.epochs - 10) else None
    train_loader, test_loader = build_dataloaders(max_val_patches=val_cap, batch_size=chosen_bs, patches_root=current_root)
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            print("GPU:", name, flush=True)
        except Exception:
            pass
    print(type(train_loader.dataset), flush=True)
    print("Train samples:", len(train_loader.dataset), flush=True)
    print("Val samples:", len(test_loader.dataset), flush=True)
    print("Batch size:", train_loader.batch_size, flush=True)
    print("num_workers:", train_loader.num_workers, flush=True)
    mode_name = type(train_loader.dataset).__name__
    if mode_name == "PatchChangeDetectionDataset":
        print("Patch mode: pre-extracted patches (whu_patches)", flush=True)
    else:
        print("Patch mode: sliding window dataset", flush=True)
    print("Train batches:", len(train_loader), flush=True)
    eff_bs = train_loader.batch_size * int(getattr(config, "grad_accum_steps", 1))
    steps_per_epoch = len(train_loader) // int(getattr(config, "grad_accum_steps", 1))
    print("Effective batch size:", eff_bs, flush=True)
    print("Optimizer steps per epoch:", steps_per_epoch, flush=True)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "whu_training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou\n")
    epoch_times = []
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch=epoch)
        scheduler.step()
        p = r = f1 = iou = None
        run_val = (epoch + 1) % 5 == 0
        if run_val:
            p, r, f1, iou = evaluate(model, test_loader, device)
            print("epoch", epoch + 1, "train_loss", round(loss, 4), "precision", round(p, 4), "recall", round(r, 4), "f1", round(f1, 4), "iou", round(iou, 4), flush=True)
        else:
            print("epoch", epoch + 1, "train_loss", round(loss, 4), flush=True)
        with open(log_path, "a") as f:
            if f1 is None:
                f.write(f"{epoch + 1},{loss},,,,\n")
            else:
                f.write(f"{epoch + 1},{loss},{p},{r},{f1},{iou}\n")
        epoch_time = time.time() - epoch_start
        print("epoch_time", round(epoch_time, 2), flush=True)
        if epoch + 1 == 20:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_iou": best_iou,
            }, os.path.join("checkpoints", "lsnet_whu_phase1.pth"))
            print("Switching to Phase 2: rebuilding patches with stride=128. This may take several minutes. Do not stop the process.", flush=True)
            ensure_patches(128, phase2_root)
            print("Phase 2 patch preparation completed. Resuming training...", flush=True)
            val_cap = 3000 if (epoch + 1) < (config.epochs - 10) else None
            train_loader, test_loader = build_dataloaders(max_val_patches=val_cap, batch_size=chosen_bs, patches_root=phase2_root)
            current_root = phase2_root
            print("Switched to stride 128 dataset", flush=True)
        if epoch + 1 == (config.epochs - 10):
            _, test_loader = build_dataloaders(max_val_patches=None, batch_size=chosen_bs, patches_root=current_root)
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_iou": best_iou,
        }, os.path.join("checkpoints", "phase2_last.pth"))
        if iou is not None and iou > best_iou:
            best_iou = iou
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "f1": f1,
                "precision": p,
                "recall": r,
                "iou": iou,
            }, os.path.join("checkpoints", "phase2_best.pth"))

if __name__ == "__main__":
    main()
