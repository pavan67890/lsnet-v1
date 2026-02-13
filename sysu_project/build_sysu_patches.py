import os
import math
import random
from PIL import Image
import numpy as np

from sysu_config import DATA_ROOT, PATCH_ROOT, PATCH_SIZE, STRIDE

def _ensure_dirs():
    for split in ["train", "val"]:
        base = os.path.join(PATCH_ROOT, split)
        for d in ["t1", "t2", "mask"]:
            os.makedirs(os.path.join(base, d), exist_ok=True)
        # optional A/B mirrors for human inspection
        for d in ["A", "B"]:
            os.makedirs(os.path.join(base, d), exist_ok=True)

def _list_pairs(root):
    candidates = []
    # common layouts:
    # root/train/A, root/train/B, root/train/label
    # root/train/time1, root/train/time2, root/train/label
    for subset in ["train", "val", "Train", "Val", "training", "validation"]:
        base = os.path.join(root, subset)
        options = [
            ("A", "B", "label"),
            ("time1", "time2", "label"),
            ("t1", "t2", "label"),
        ]
        for a_name, b_name, l_name in options:
            a_dir = os.path.join(base, a_name)
            b_dir = os.path.join(base, b_name)
            l_dir = os.path.join(base, l_name)
            if os.path.isdir(a_dir) and os.path.isdir(b_dir) and os.path.isdir(l_dir):
                names = sorted([f for f in os.listdir(a_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))])
                for n in names:
                    an = os.path.join(a_dir, n)
                    bn = os.path.join(b_dir, n)
                    ln = os.path.join(l_dir, n)
                    if os.path.exists(bn) and os.path.exists(ln):
                        candidates.append((an, bn, ln))
                break
    return candidates

def _list_pairs_subset(root, subset):
    base = os.path.join(root, subset)
    candidates = []
    options = [
        ("A", "B", "label"),
        ("time1", "time2", "label"),
        ("t1", "t2", "label"),
    ]
    for a_name, b_name, l_name in options:
        a_dir = os.path.join(base, a_name)
        b_dir = os.path.join(base, b_name)
        l_dir = os.path.join(base, l_name)
        if os.path.isdir(a_dir) and os.path.isdir(b_dir) and os.path.isdir(l_dir):
            names = sorted([f for f in os.listdir(a_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))])
            for n in names:
                an = os.path.join(a_dir, n)
                bn = os.path.join(b_dir, n)
                ln = os.path.join(l_dir, n)
                if os.path.exists(bn) and os.path.exists(ln):
                    candidates.append((an, bn, ln))
            break
    return candidates

def _load_img(path):
    im = Image.open(path).convert("RGB")
    arr = np.array(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    return arr

def _load_mask(path):
    im = Image.open(path).convert("L")
    arr = (np.array(im, dtype=np.float32) > 127).astype(np.float32)
    # mask as 1xH x W
    return arr

def _make_patches(t1, t2, m, size=256, stride=128):
    C, H, W = t1.shape
    ph = size
    pw = size
    patches = []
    for y in range(0, max(1, H - ph + 1), stride):
        for x in range(0, max(1, W - pw + 1), stride):
            t1p = t1[:, y:y+ph, x:x+pw]
            t2p = t2[:, y:y+ph, x:x+pw]
            mp = m[y:y+ph, x:x+pw]
            if t1p.shape[-2:] != (ph, pw) or mp.shape != (ph, pw):
                continue
            patches.append((t1p, t2p, mp))
    return patches

def main():
    os.makedirs(PATCH_ROOT, exist_ok=True)
    _ensure_dirs()
    train_pairs = _list_pairs_subset(DATA_ROOT, "train")
    val_pairs = _list_pairs_subset(DATA_ROOT, "val")
    if len(train_pairs) == 0 and len(val_pairs) == 0:
        pairs = _list_pairs(DATA_ROOT)
        if len(pairs) == 0:
            print("No SYSU pairs found under", DATA_ROOT)
            return
        random.seed(42)
        random.shuffle(pairs)
        k = int(0.8 * len(pairs))
        train_pairs = pairs[:k]
        val_pairs = pairs[k:]
    def process(split_pairs, split):
        base = os.path.join(PATCH_ROOT, split)
        t1_dir = os.path.join(base, "t1")
        t2_dir = os.path.join(base, "t2")
        m_dir = os.path.join(base, "mask")
        a_dir = os.path.join(base, "A")
        b_dir = os.path.join(base, "B")
        idx = 0
        for a, b, l in split_pairs:
            t1 = _load_img(a)
            t2 = _load_img(b)
            m = _load_mask(l)
            patches = _make_patches(t1, t2, m, size=PATCH_SIZE, stride=STRIDE)
            for t1p, t2p, mp in patches:
                fname = f"{split}_{idx}.npy"
                np.save(os.path.join(t1_dir, fname), t1p.astype(np.float16))
                np.save(os.path.join(t2_dir, fname), t2p.astype(np.float16))
                np.save(os.path.join(m_dir, fname), mp.astype(np.uint8))
                idx += 1
        print(split, "patches saved:", idx)
    process(train_pairs, "train")
    process(val_pairs, "val")
    print("SYSU patches build complete")

if __name__ == "__main__":
    main()
