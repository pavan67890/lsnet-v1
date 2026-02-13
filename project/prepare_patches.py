import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import argparse
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=getattr(config, "train_stride", 128))
    parser.add_argument("--patch_size", type=int, default=getattr(config, "patch_size", 256))
    parser.add_argument("--dest", type=str, default=os.path.join(os.path.dirname(getattr(config, "data_root", config.t1_dir)), "whu_patches"))
    args = parser.parse_args()
    src_t1 = config.t1_dir
    src_t2 = config.t2_dir
    src_mask = config.mask_dir
    root = args.dest
    t1_dir = os.path.join(root, "t1")
    t2_dir = os.path.join(root, "t2")
    m_dir = os.path.join(root, "mask")
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(m_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(src_t1) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    ps = int(args.patch_size)
    s = int(args.stride)
    pid = 1
    total_patches = 0
    log_path = os.path.join(os.path.dirname(root), "patch_prep_log.txt")
    errors = 0
    for idx, fname in enumerate(files, 1):
        try:
            t1 = Image.open(os.path.join(src_t1, fname)).convert("RGB")
            t2 = Image.open(os.path.join(src_t2, fname)).convert("RGB")
        except Exception:
            errors += 1
            continue
        base, ext = os.path.splitext(fname)
        m_path = os.path.join(src_mask, fname)
        if not os.path.exists(m_path):
            for e in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                q = os.path.join(src_mask, base + e)
                if os.path.exists(q):
                    m_path = q
                    break
        try:
            m = Image.open(m_path).convert("L")
        except Exception:
            errors += 1
            continue
        w, h = m.size
        for y in range(0, max(1, h - ps + 1), s):
            for x in range(0, max(1, w - ps + 1), s):
                t1p = TF.to_tensor(TF.crop(t1, y, x, ps, ps)).numpy().astype(np.float32)
                t2p = TF.to_tensor(TF.crop(t2, y, x, ps, ps)).numpy().astype(np.float32)
                mp = TF.to_tensor(TF.crop(m, y, x, ps, ps)).numpy().astype(np.float32)
                name = f"patch_{pid:06d}.npy"
                np.save(os.path.join(t1_dir, name), t1p)
                np.save(os.path.join(t2_dir, name), t2p)
                np.save(os.path.join(m_dir, name), mp)
                pid += 1
                total_patches += 1
        if idx % 100 == 0:
            msg = f"processed_images {idx} total_patches {total_patches} errors {errors}\n"
            print(msg.strip(), flush=True)
            try:
                with open(log_path, "a") as lf:
                    lf.write(msg)
            except Exception:
                pass
    msg = f"final_total_patches {total_patches} errors {errors}\n"
    print(msg.strip(), flush=True)
    try:
        with open(log_path, "a") as lf:
            lf.write(msg)
    except Exception:
        pass
    ct1 = len(os.listdir(t1_dir))
    ct2 = len(os.listdir(t2_dir))
    cm = len(os.listdir(m_dir))
    ok = (ct1 == ct2 == cm)
    msg = f"counts {ct1} {ct2} {cm} match {ok}\n"
    print(msg.strip(), flush=True)
    try:
        with open(log_path, "a") as lf:
            lf.write(msg)
    except Exception:
        pass

if __name__ == "__main__":
    main()
