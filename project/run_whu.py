import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from models.lsnet import LSNet
from losses.bce_dice import BCEDiceLoss
from utils import WarmupCosine, count_params
import train as T
import config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    print("starting_epochs", int(config.epochs))
    print("encoder_params", count_params(model.encoder))
    print("total_params", count_params(model))
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = WarmupCosine(optimizer, warmup_epochs=config.warmup_epochs, total_epochs=config.epochs)
    scaler = GradScaler()
    train_ds = T.ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="train", patch_size=config.patch_size, stride=config.train_stride)
    val_ds = T.ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="test", patch_size=config.patch_size, stride=config.test_stride)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=config.micro_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.micro_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print("train_len", len(train_loader.dataset), "val_len", len(val_loader.dataset))
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "whu_training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,precision,recall,f1,iou\n")
    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = -1.0
    epoch_times = []
    print("begin_loop")
    try:
        for epoch in range(config.epochs):
            t0 = time.time()
            loss = T.train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
            scheduler.step()
            p, r, f1, iou = T.evaluate(model, val_loader, device)
            print("epoch", epoch + 1, "loss", round(loss, 4), "p", round(p, 4), "r", round(r, 4), "f1", round(f1, 4), "iou", round(iou, 4))
            with open(log_path, "a") as f:
                f.write(f"{epoch + 1},{loss},{p},{r},{f1},{iou}\n")
            epoch_times.append(time.time() - t0)
            if epoch + 1 == 2:
                est = sum(epoch_times) / len(epoch_times)
                print("est_epoch_time_sec", round(est, 2))
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "f1": f1,
                    "precision": p,
                    "recall": r,
                    "iou": iou,
                }, os.path.join("checkpoints", "lsnet_whu_best.pth"))
    except Exception as e:
        print("train_error", str(e))

if __name__ == "__main__":
    main()
