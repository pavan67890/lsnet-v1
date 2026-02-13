import os
import csv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.lsnet import LSNet
from datasets.cd_dataset import PatchChangeDetectionDataset
import sysu_config as config

def _load_model(device):
    model = LSNet().to(device)
    ckpt = os.path.join(config.CHECKPOINT_DIR, "sysu_best.pth")
    payload = torch.load(ckpt, map_location=device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    print("Loaded checkpoint path", ckpt)
    print("Epoch", payload.get("epoch", None))
    print("best_thr", payload.get("best_thr", None))
    return model, float(payload.get("best_thr", 0.88))

def _build_loader(split="val"):
    root = os.path.join(config.PATCH_ROOT, split)
    ds = PatchChangeDetectionDataset(root, mode="test" if split!="train" else "train")
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=getattr(config, "BATCH_SIZE", 8), shuffle=False, num_workers=getattr(config, "NUM_WORKERS", 2), pin_memory=True, persistent_workers=getattr(config, "NUM_WORKERS", 2)>0, prefetch_factor=2 if getattr(config, "NUM_WORKERS", 2)>0 else None)

def global_threshold_search(model, device, start=0.80, end=0.95, step=0.005):
    vloader = _build_loader("val")
    th = torch.arange(start, end + 1e-8, step, device=device)
    tp = torch.zeros_like(th, dtype=torch.float64); fp = torch.zeros_like(th, dtype=torch.float64); fn = torch.zeros_like(th, dtype=torch.float64)
    with torch.no_grad():
        for t1, t2, m in vloader:
            t1 = t1.to(device); t2 = t2.to(device); m = m.to(device)
            logits, probs = model(t1, t2)
            if m.max().item() > 1.0: m = m / 255.0
            pos = (m > 0.5); neg = ~pos
            for i in range(th.numel()):
                pred = probs > th[i]
                tp[i] += (pred & pos).sum()
                fp[i] += (pred & neg).sum()
                fn[i] += ((~pred) & pos).sum()
    eps = 1e-8
    precision = tp/(tp+fp+eps); recall = tp/(tp+fn+eps); f1 = 2*precision*recall/(precision+recall+eps); iou = tp/(tp+fp+fn+eps)
    best_idx = int(torch.argmax(iou).item())
    best_thr = float(th[best_idx].item()); best_iou = float(iou[best_idx].item())
    out = os.path.join(config.LOG_DIR, "sysu_threshold_search.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["threshold","precision","recall","f1","iou"])
        for i in range(th.numel()):
            w.writerow([float(th[i].item()), float(precision[i].item()), float(recall[i].item()), float(f1[i].item()), float(iou[i].item())])
    with open(os.path.join(config.LOG_DIR, "sysu_best_threshold.txt"), "w") as f:
        f.write(str(best_thr))
    print("Best threshold", best_thr, "Best IoU", best_iou)
    return best_thr, (precision[best_idx].item(), recall[best_idx].item(), f1[best_idx].item(), iou[best_idx].item())

def tta_eval(model, device, thr):
    vloader = _build_loader("val")
    tp_n=fp_n=fn_n=0.0; tp_t=fp_t=fn_t=0.0
    with torch.no_grad():
        for t1,t2,m in vloader:
            t1=t1.to(device); t2=t2.to(device); m=m.to(device)
            logits, probs = model(t1,t2)
            if m.max().item()>1.0: m=m/255.0
            gt=(m>0.5)
            pred_n = (probs>thr)
            tp_n += (pred_n & gt).sum().item(); fp_n += (pred_n & (~gt)).sum().item(); fn_n += ((~pred_n) & gt).sum().item()
            t1h=torch.flip(t1,[-1]); t2h=torch.flip(t2,[-1]); lh,ph=model(t1h,t2h); ph=torch.flip(ph,[-1])
            t1v=torch.flip(t1,[-2]); t2v=torch.flip(t2,[-2]); lv,pv=model(t1v,t2v); pv=torch.flip(pv,[-2])
            t1hv=torch.flip(torch.flip(t1,[-1]),[-2]); t2hv=torch.flip(torch.flip(t2,[-1]),[-2]); lhv,phv=model(t1hv,t2hv); phv=torch.flip(torch.flip(phv,[-1]),[-2])
            p_avg=(probs+ph+pv+phv)/4.0
            pred_t=(p_avg>thr)
            tp_t += (pred_t & gt).sum().item(); fp_t += (pred_t & (~gt)).sum().item(); fn_t += ((~pred_t) & gt).sum().item()
    eps=1e-8
    def stats(tp,fp,fn):
        prec = tp/(tp+fp+eps); rec = tp/(tp+fn+eps); f1 = 2*prec*rec/(prec+rec+eps); iou = tp/(tp+fp+fn+eps); return prec,rec,f1,iou
    normal = stats(tp_n,fp_n,fn_n); tta = stats(tp_t,fp_t,fn_t)
    out = os.path.join(config.LOG_DIR, "sysu_tta_results.txt")
    with open(out, "w") as f:
        f.write(f"Threshold used: {thr}\n")
        f.write(f"Normal: Precision {normal[0]} Recall {normal[1]} F1 {normal[2]} IoU {normal[3]}\n")
        f.write(f"TTA: Precision {tta[0]} Recall {tta[1]} F1 {tta[2]} IoU {tta[3]}\n")
        f.write(f"IoU gain {tta[3]-normal[3]}\n")
        f.write(f"F1 gain {tta[2]-normal[2]}\n")
    return normal, tta

def plots():
    os.makedirs(os.path.join(config.LOG_DIR, "sysu_plots"), exist_ok=True)
    import csv
    th=[]; iou=[]; prec=[]; rec=[]
    with open(os.path.join(config.LOG_DIR, "sysu_threshold_search.csv"), "r") as f:
        r = csv.reader(f); next(r)
        for row in r:
            th.append(float(row[0])); prec.append(float(row[1])); rec.append(float(row[2])); iou.append(float(row[4]))
    plt.figure(figsize=(6,5)); plt.plot(th,iou); plt.xlabel("Threshold"); plt.ylabel("IoU"); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(config.LOG_DIR, "sysu_plots", "threshold_vs_iou.png"), dpi=300); plt.close()
    plt.figure(figsize=(6,5)); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(config.LOG_DIR, "sysu_plots", "precision_recall_curve.png"), dpi=300); plt.close()
    # confusion matrix (normal at best thr)
    # For simplicity reuse global stats and cannot recompute TN precisely without extra pass; skip detailed TN visualization here.
    from PIL import Image
    # Placeholder bar chart
    with open(os.path.join(config.LOG_DIR, "sysu_tta_results.txt"), "r") as f:
        lines = f.read().splitlines()
    import re
    def ex(s): return [float(x) for x in re.findall(r'-?\\d+\\.\\d+', s)]
    n = ex([l for l in lines if l.startswith("Normal:")][0]); t = ex([l for l in lines if l.startswith("TTA:")][0])
    labels=['Normal','TTA']; import numpy as np; x=np.arange(2); w=0.35
    plt.figure(figsize=(6,4)); plt.bar(x-w/2,[n[3], t[3]],width=w,label='IoU'); plt.bar(x+w/2,[n[2], t[2]],width=w,label='F1'); plt.xticks(x,labels); plt.ylim(0.0,1.0); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(config.LOG_DIR, "sysu_plots", "normal_vs_tta_bar.png"), dpi=300); plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, thr_ckpt = _load_model(device)
    best_thr, normal = global_threshold_search(model, device, 0.80, 0.95, 0.005)
    normal_stats, tta_stats = tta_eval(model, device, best_thr)
    plots()

if __name__ == "__main__":
    main()
