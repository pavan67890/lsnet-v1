import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import config

def prepare_split(split_name):
    src_root = os.path.join(config.DATA_ROOT, split_name)
    t1_dir = os.path.join(src_root, "A")
    t2_dir = os.path.join(src_root, "B")
    m_dir = os.path.join(src_root, "label")
    out_root = os.path.join(config.PATCHES_ROOT, split_name)
    ot1 = os.path.join(out_root, "t1"); ot2 = os.path.join(out_root, "t2"); om = os.path.join(out_root, "mask")
    os.makedirs(ot1, exist_ok=True); os.makedirs(ot2, exist_ok=True); os.makedirs(om, exist_ok=True)
    try:
        files = sorted([f for f in os.listdir(t1_dir) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))])
    except Exception:
        files = []
    ps = config.patch_size
    if split_name == "train":
        stride = config.train_stride
    elif split_name == "val":
        stride = config.val_stride
    else:
        stride = config.test_stride
    pid = 1
    print(f"Preparing {split_name}…")
    for idx, fname in enumerate(files, 1):
        t1 = Image.open(os.path.join(t1_dir, fname)).convert("RGB")
        t2 = Image.open(os.path.join(t2_dir, fname)).convert("RGB")
        base, _ = os.path.splitext(fname)
        m_path = os.path.join(m_dir, fname)
        if not os.path.exists(m_path):
            for e in [".png",".jpg",".jpeg",".tif",".tiff"]:
                q = os.path.join(m_dir, base + e)
                if os.path.exists(q):
                    m_path = q; break
        m = Image.open(m_path).convert("L")
        w,h = m.size
        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(0, max(1, w - ps + 1), stride):
                t1p = TF.to_tensor(TF.crop(t1, y, x, ps, ps)).numpy().astype(np.float32)
                t2p = TF.to_tensor(TF.crop(t2, y, x, ps, ps)).numpy().astype(np.float32)
                mp = TF.to_tensor(TF.crop(m, y, x, ps, ps)).numpy().astype(np.float32)
                name = f"{split_name}_patch_{pid:06d}.npy"
                np.save(os.path.join(ot1, name), t1p)
                np.save(os.path.join(ot2, name), t2p)
                np.save(os.path.join(om, name), mp)
                pid += 1

def main():
    os.makedirs(config.PATCHES_ROOT, exist_ok=True)
    for split in ["train","val","test"]:
        base = os.path.join(config.PATCHES_ROOT, split)
        os.makedirs(os.path.join(base, "t1"), exist_ok=True)
        os.makedirs(os.path.join(base, "t2"), exist_ok=True)
        os.makedirs(os.path.join(base, "mask"), exist_ok=True)
        prepare_split(split)
    # counts
    def count_split(split):
        mdir = os.path.join(config.PATCHES_ROOT, split, "mask")
        return len([f for f in os.listdir(mdir) if f.lower().endswith(".npy")]) if os.path.isdir(mdir) else 0
    tr = count_split("train"); va = count_split("val"); te = count_split("test")
    print("Prepared patches at", config.PATCHES_ROOT)
    print("Train patches:", tr)
    print("Val patches:", va)
    print("Test patches:", te)

if __name__ == "__main__":
    main()
