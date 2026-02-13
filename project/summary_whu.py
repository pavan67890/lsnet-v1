import os
from PIL import Image
import numpy as np
import config

def summary(root=None):
    t1 = config.t1_dir if root is None else os.path.join(root, "t1")
    t2 = config.t2_dir if root is None else os.path.join(root, "t2")
    m = config.mask_dir if root is None else os.path.join(root, "mask")
    files = sorted([f for f in os.listdir(t1) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    sizes = []
    uniq = set()
    count_pairs = 0
    for fname in files:
        p1 = os.path.join(t1, fname)
        p2 = os.path.join(t2, fname)
        base, ext = os.path.splitext(fname)
        pm = os.path.join(m, fname)
        if not os.path.exists(pm):
            for e in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                q = os.path.join(m, base + e)
                if os.path.exists(q):
                    pm = q
                    break
        if not (os.path.exists(p1) and os.path.exists(p2) and os.path.exists(pm)):
            continue
        img = Image.open(p1).convert("RGB")
        sizes.append(img.size)
        try:
            ms = Image.open(pm).convert("L")
            arr = np.array(ms)
            uniq.update(np.unique(arr).tolist())
            count_pairs += 1
        except Exception:
            continue
    if sizes:
        wmin = min(s[0] for s in sizes)
        wmax = max(s[0] for s in sizes)
        hmin = min(s[1] for s in sizes)
        hmax = max(s[1] for s in sizes)
        print("pairs", count_pairs)
        print("size_range", (wmin, hmin), (wmax, hmax))
        print("mask_values", sorted(list(uniq)))
        try:
            with open("s:\\sota\\whu_summary.txt", "w") as f:
                f.write(f"pairs {count_pairs}\n")
                f.write(f"size_range {(wmin, hmin)} {(wmax, hmax)}\n")
                f.write(f"mask_values {sorted(list(uniq))}\n")
        except Exception:
            pass
    else:
        print("pairs", 0)
        print("size_range", None)
        print("mask_values", None)

if __name__ == "__main__":
    summary()
