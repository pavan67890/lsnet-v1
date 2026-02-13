import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
from utils.utils import metrics
import config

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

def remove_small_components(mask, min_size=16):
    m = mask.detach().cpu().numpy()
    B = m.shape[0]
    out = []
    for b in range(B):
        arr = (m[b,0] > 0.5).astype(np.uint8)
        H,W = arr.shape
        vis = np.zeros_like(arr, dtype=np.uint8)
        def neigh(y,x):
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                ny,nx = y+dy,x+dx
                if 0<=ny<H and 0<=nx<W:
                    yield ny,nx
        for y in range(H):
            for x in range(W):
                if arr[y,x]==0 or vis[y,x]==1: continue
                stack=[(y,x)]; comp=[]
                vis[y,x]=1
                while stack:
                    cy,cx = stack.pop(); comp.append((cy,cx))
                    for ny,nx in neigh(cy,cx):
                        if arr[ny,nx]==1 and vis[ny,nx]==0:
                            vis[ny,nx]=1; stack.append((ny,nx))
                if len(comp) < min_size:
                    for cy,cx in comp:
                        arr[cy,cx]=0
        out.append(arr.astype(np.float32))
    out = torch.from_numpy(np.stack(out, axis=0)).unsqueeze(1)
    return out.to(mask.device)

def morph_close(mask, k=3):
    import torch.nn.functional as F
    pad = k//2
    x = (mask > 0.5).float()
    dil = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
    ones = F.avg_pool2d(dil, kernel_size=k, stride=1, padding=pad) * (k*k)
    closed = (ones == (k*k)).float()
    return closed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    cp = os.path.join(config.CHECKPOINTS_DIR, "levir_best.pth")
    payload = torch.load(cp, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    ds = PatchChangeDetectionDataset(os.path.join(config.PATCHES_ROOT, "test"), mode="test")
    loader = DataLoader(ds, batch_size=config.micro_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    base = os.path.join("S:/sota/levir_project/outputs")
    os.makedirs(base, exist_ok=True)
    # final threshold; find best via stored or search
    thr = 0.34
    p_sum = r_sum = f1_sum = iou_sum = 0.0; n=0
    with torch.no_grad():
        for t1, t2, m in loader:
            t1 = t1.to(device, non_blocking=True); t2 = t2.to(device, non_blocking=True); m = m.to(device, non_blocking=True)
            probs = tta_predict(model, t1, t2, device)
            pred = (probs > thr).float()
            pred = remove_small_components(pred, min_size=16)
            pred = morph_close(pred, k=3)
            p,r,f1,iou = metrics(pred, m, threshold=0.5)
            p_sum += p; r_sum += r; f1_sum += f1; iou_sum += iou; n+=1
    precision = p_sum/n; recall = r_sum/n; f1 = f1_sum/n; iou = iou_sum/n
    with open(os.path.join(base, "final_metrics.txt"), "w") as f:
        f.write(f"Precision: {precision:.6f}\n")
        f.write(f"Recall: {recall:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        f.write(f"IoU: {iou:.6f}\n")
    print("Saved final metrics to", os.path.join(base, "final_metrics.txt"))

if __name__ == "__main__":
    with torch.no_grad():
        main()
