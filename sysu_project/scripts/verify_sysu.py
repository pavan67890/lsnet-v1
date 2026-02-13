import os
import numpy as np

BASE = r"S:\sota\sysu_project\sysu_patches"

def count_npy(p):
    if not os.path.isdir(p):
        return 0
    return len([f for f in os.listdir(p) if f.lower().endswith(".npy")])

def main():
    train_t1 = os.path.join(BASE, "train", "t1")
    train_t2 = os.path.join(BASE, "train", "t2")
    train_m = os.path.join(BASE, "train", "mask")
    val_t1 = os.path.join(BASE, "val", "t1")
    val_t2 = os.path.join(BASE, "val", "t2")
    val_m = os.path.join(BASE, "val", "mask")
    exists_all = all(os.path.isdir(p) for p in [train_t1,train_t2,train_m,val_t1,val_t2,val_m])
    ct1 = count_npy(train_t1); ct2 = count_npy(train_t2); cm = count_npy(train_m)
    vt1 = count_npy(val_t1); vt2 = count_npy(val_t2); vm = count_npy(val_m)
    print("Exists:", exists_all)
    print("Train .npy counts: t1", ct1, "t2", ct2, "mask", cm)
    print("Val .npy counts:   t1", vt1, "t2", vt2, "mask", vm)
    train_count = min(ct1, ct2, cm)
    val_count = min(vt1, vt2, vm)
    chg_total = 0.0; px_total = 0.0
    if os.path.isdir(train_m):
        for f in os.listdir(train_m):
            if f.lower().endswith(".npy"):
                m = np.load(os.path.join(train_m, f)).astype(np.float32)
                chg_total += (m > 0.5).sum()
                px_total += m.size
    pct = (chg_total / px_total * 100.0) if px_total > 0 else 0.0
    print("number of train patches:", train_count)
    print("number of val patches:", val_count)
    print("percentage of change pixels (train):", pct)
    # compute val change percentage
    vchg = 0.0; vpx = 0.0
    if os.path.isdir(val_m):
        for f in os.listdir(val_m):
            if f.lower().endswith(".npy"):
                m = np.load(os.path.join(val_m, f)).astype(np.float32)
                vchg += (m > 0.5).sum()
                vpx += m.size
    vpct = (vchg / vpx * 100.0) if vpx > 0 else 0.0
    print("percentage of change pixels (val):", vpct)

if __name__ == "__main__":
    main()
