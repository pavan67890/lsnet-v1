import os
import shutil
from PIL import Image
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def convert_mask(img):
    img = img.convert("L")
    arr = np.array(img)
    arr = np.where(arr > 0, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")

def prepare(src_root, dst_root):
    alt_root = os.path.join(src_root, "1. The two-period image data")
    t1_dir = os.path.join(dst_root, "t1")
    t2_dir = os.path.join(dst_root, "t2")
    m_dir = os.path.join(dst_root, "mask")
    ensure_dir(t1_dir)
    ensure_dir(t2_dir)
    ensure_dir(m_dir)
    if os.path.isdir(alt_root):
        yr1 = os.path.join(alt_root, "2012", "splited_images")
        yr2 = os.path.join(alt_root, "2016", "splited_images")
        for split in ("train", "test"):
            a_dir = os.path.join(yr1, split, "image")
            b_dir = os.path.join(yr2, split, "image")
            l_dir = os.path.join(yr1, split, "label")
            if not (os.path.isdir(a_dir) and os.path.isdir(b_dir) and os.path.isdir(l_dir)):
                continue
            files = sorted([f for f in os.listdir(a_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
            for fname in files:
                a_path = os.path.join(a_dir, fname)
                b_path = os.path.join(b_dir, fname)
                l_path = os.path.join(l_dir, fname)
                if not (os.path.exists(a_path) and os.path.exists(b_path) and os.path.exists(l_path)):
                    continue
                Image.open(a_path).convert("RGB").save(os.path.join(t1_dir, fname))
                Image.open(b_path).convert("RGB").save(os.path.join(t2_dir, fname))
                mask = Image.open(l_path)
                mask = convert_mask(mask)
                mask.save(os.path.join(m_dir, fname))
    else:
        a_dir = os.path.join(src_root, "A")
        b_dir = os.path.join(src_root, "B")
        l_dir = os.path.join(src_root, "label")
        files = sorted([f for f in os.listdir(a_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]) if os.path.isdir(a_dir) else []
        for fname in files:
            a_path = os.path.join(a_dir, fname)
            b_path = os.path.join(b_dir, fname)
            l_path = os.path.join(l_dir, fname)
            if not (os.path.exists(a_path) and os.path.exists(b_path) and os.path.exists(l_path)):
                continue
            Image.open(a_path).convert("RGB").save(os.path.join(t1_dir, fname))
            Image.open(b_path).convert("RGB").save(os.path.join(t2_dir, fname))
            mask = Image.open(l_path)
            mask = convert_mask(mask)
            mask.save(os.path.join(m_dir, fname))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default=os.path.join(os.path.dirname(__file__), "data", "whu"))
    args = parser.parse_args()
    prepare(args.src, args.dst)
    print("done")

if __name__ == "__main__":
    main()
