import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
from utils import set_seed, metrics
import config
from torch.optim.swa_utils import AveragedModel, update_bn

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_dataloaders(max_val_patches=3000, batch_size=4):
    root = os.path.join(os.path.dirname(config.data_root), "whu_patches")
    train_ds = PatchChangeDetectionDataset(root, mode="train")
    val_ds = PatchChangeDetectionDataset(root, mode="test")
    if max_val_patches is not None:
        n = len(val_ds)
        if n > max_val_patches:
            idx = random.sample(range(n), max_val_patches)
            val_ds = Subset(val_ds, idx)
    workers = 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, grad_accum_steps=2, epoch=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    desc = f"Train Epoch {epoch + 1}" if epoch is not None else "Train"
    for i, (t1, t2, m) in enumerate(tqdm(loader, desc=desc, leave=False)):
        t1 = t1.to(device, non_blocking=True)
        t2 = t2.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            logits, _ = model(t1, t2)
            loss = criterion(logits, m) / grad_accum_steps
        if torch.isnan(loss).any():
            print("NaN loss detected — stopping epoch")
            return 0.0
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * grad_accum_steps * t1.size(0)
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
    cp_path = r"S:\sota\project\checkpoints\lsnet_whu_best.pth"
    if not os.path.exists(cp_path):
        print("Checkpoint not found. Abort.")
        return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        print("Checkpoint missing model state_dict. Abort.")
        return
    model.load_state_dict(state)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=config.weight_decay)
    scaler = GradScaler(device='cuda')
    swa_model = AveragedModel(model)
    grad_accum_steps = 2
    bs = 4
    train_loader, val_loader = build_dataloaders(max_val_patches=3000, batch_size=bs)
    print("Batch size:", bs)
    print("num_workers:", train_loader.num_workers)
    print("Train batches:", len(train_loader))
    print("Effective batch size:", bs * grad_accum_steps)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "whu_finetune_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou\n")
    from losses.focal_tversky import FocalTverskyLoss
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
    best_iou = -1.0
    best_f1 = -1.0
    for epoch in range(0, 20):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, grad_accum_steps=grad_accum_steps, epoch=epoch)
        swa_model.update_parameters(model)
        p = r = f1 = iou = None
        if (epoch + 1) % 5 == 0:
            p, r, f1, iou = evaluate(model, val_loader, device)
            print("epoch", epoch + 1, "train_loss", round(train_loss, 4), "precision", round(p, 4), "recall", round(r, 4), "f1", round(f1, 4), "iou", round(iou, 4))
        else:
            print("epoch", epoch + 1, "train_loss", round(train_loss, 4))
        with open(log_path, "a") as f:
            if f1 is None:
                f.write(f"{epoch + 1},{train_loss},,,,\n")
            else:
                f.write(f"{epoch + 1},{train_loss},{p},{r},{f1},{iou}\n")
        if f1 is not None:
            better = (iou > best_iou) or (iou == best_iou and f1 > best_f1)
            if better:
                best_iou = iou
                best_f1 = f1
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "f1": f1,
                    "precision": p,
                    "recall": r,
                    "iou": iou,
                }, os.path.join("checkpoints", "lsnet_whu_finetune_best.pth"))
        print("epoch_time", round(time.time() - t0, 2))
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }, os.path.join("checkpoints", "lsnet_whu_finetune_last.pth"))
    update_bn(train_loader, swa_model)
    torch.save({
        "model": swa_model.module.state_dict() if hasattr(swa_model, "module") else swa_model.state_dict(),
        "epoch": 20,
        "best_iou": best_iou,
        "best_f1": best_f1,
    }, os.path.join("checkpoints", "lsnet_whu_swa_final.pth"))

if __name__ == "__main__":
    main()
