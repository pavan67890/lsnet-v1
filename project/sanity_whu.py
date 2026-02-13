import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import config
from datasets.cd_dataset import ChangeDetectionDataset
from models.lsnet import LSNet
from losses.bce_dice import BCEDiceLoss
from utils import count_params, set_seed, metrics

def mask_change_ratio_for_patch(mask_dir, fname, x, y, ps):
    m = Image.open(os.path.join(mask_dir, fname)).convert("L")
    mp = m.crop((x, y, x + ps, y + ps))
    arr = np.array(mp)
    change_pixels = (arr > 0).sum()
    total_pixels = ps * ps
    return change_pixels / float(total_pixels)

def sanity():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not (os.path.isdir(config.t1_dir) and os.path.isdir(config.t2_dir) and os.path.isdir(config.mask_dir)):
        print("dataset_missing")
        return
    logs = []
    ds = ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="train", patch_size=config.patch_size, stride=config.train_stride)
    n_patches = len(ds)
    ch = 0
    nch = 0
    for fname, x, y in ds.index_map[: min(5000, len(ds.index_map))]:
        r = mask_change_ratio_for_patch(config.mask_dir, fname, x, y, config.patch_size)
        if r == 0.0:
            nch += 1
        else:
            ch += 1
    print("patches", n_patches)
    print("change_vs_nochange", ch, nch)
    logs.append(f"patches {n_patches}")
    logs.append(f"change_vs_nochange {ch} {nch}")
    loader = DataLoader(ds, batch_size=config.micro_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    t1, t2, m = next(iter(loader))
    print("batch_shapes", t1.shape, t2.shape, m.shape)
    mv = torch.unique((m > 0.5).float()).cpu().numpy().tolist()
    print("mask_unique", mv)
    logs.append(f"batch_shapes {tuple(t1.shape)} {tuple(t2.shape)} {tuple(m.shape)}")
    logs.append(f"mask_unique {mv}")
    model = LSNet().to(device)
    criterion = BCEDiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler()
    t1 = t1.to(device, non_blocking=True)
    t2 = t2.to(device, non_blocking=True)
    m = m.to(device, non_blocking=True)
    torch.cuda.reset_peak_memory_stats(device)
    with autocast():
        logits, probs = model(t1, t2)
        loss = criterion(logits, m)
    print("logits_shape", logits.shape)
    print("probs_minmax", float(probs.min().item()), float(probs.max().item()))
    logs.append(f"logits_shape {tuple(logits.shape)}")
    logs.append(f"probs_minmax {float(probs.min().item())} {float(probs.max().item())}")
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    mm = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print("max_mem_gb", mm)
    logs.append(f"max_mem_gb {mm}")
    ds_small = ChangeDetectionDataset(config.t1_dir, config.t2_dir, config.mask_dir, mode="train", patch_size=config.patch_size, stride=config.train_stride)
    files = sorted(os.listdir(config.t1_dir))[:10]
    idxs = [(f, 0, 0) for f in files]
    ds_small.index_map = idxs
    loader_small = DataLoader(ds_small, batch_size=config.micro_batch_size, shuffle=True, num_workers=0)
    for i in range(50):
        for t1, t2, m in loader_small:
            t1 = t1.to(device)
            t2 = t2.to(device)
            m = m.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits, probs = model(t1, t2)
                loss = criterion(logits, m)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if i % 10 == 0:
            print("iter", i, "loss", float(loss.item()))
            logs.append(f"iter {i} loss {float(loss.item())}")
    try:
        with open("s:\\sota\\whu_sanity.txt", "w") as f:
            f.write("\n".join(logs))
    except Exception:
        pass

if __name__ == "__main__":
    sanity()
