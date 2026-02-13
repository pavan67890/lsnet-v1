import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.cd_dataset import PatchChangeDetectionDataset
from sysu_config import PATCH_ROOT
from models.lsnet import LSNet
import numpy as np
import cv2

# --- CONFIG ---
CHECKPOINT_PATH = r"S:\sota\sysu_project\checkpoints\syu iou 63.pth"
BATCH_SIZE = 1  # TTA requires batch size 1 for safety
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_tta_predictions(model, img1, img2):
    """
    Performs 8-way Test Time Augmentation (TTA).
    Returns the averaged probability map.
    """
    preds = []
    transforms = [
        (0, None), (1, None), (2, None), (3, None),
        (0, 2), (0, 3), (1, 2), (1, 3)
    ]
    model.eval()
    with torch.no_grad():
        for k, flip in transforms:
            x1 = img1.clone()
            x2 = img2.clone()
            if flip:
                x1 = torch.flip(x1, (flip,))
                x2 = torch.flip(x2, (flip,))
            if k > 0:
                x1 = torch.rot90(x1, k, (2, 3))
                x2 = torch.rot90(x2, k, (2, 3))
            out = model(x1, x2)[0]
            prob = torch.sigmoid(out)
            if k > 0:
                prob = torch.rot90(prob, -k, (2, 3))
            if flip:
                prob = torch.flip(prob, (flip,))
            preds.append(prob)
    avg_pred = torch.stack(preds).mean(dim=0)
    # --- MORPHOLOGICAL POST-PROCESSING ---
    mask = (avg_pred > 0.5).float().cpu().numpy().astype(np.uint8)
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        mask2d = mask[0, 0]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2d = cv2.morphologyEx(mask2d, cv2.MORPH_OPEN, kernel)
        mask2d = cv2.morphologyEx(mask2d, cv2.MORPH_CLOSE, kernel)
        mask = np.expand_dims(np.expand_dims(mask2d, 0), 0)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    avg_pred = torch.from_numpy(mask).float().to(DEVICE)
    return avg_pred

def evaluate_tta():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = LSNet().to(DEVICE)
    payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    test_ds = PatchChangeDetectionDataset(os.path.join(PATCH_ROOT, "val"), mode="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    TP, TN, FP, FN = 0, 0, 0, 0
    print("Starting 8-Way TTA Evaluation...")
    for batch in tqdm(test_loader):
        img1, img2, label = batch
        img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
        label = label.to(DEVICE)
        pred_prob = get_tta_predictions(model, img1, img2)
        pred_mask = (pred_prob > 0.5).float()
        tp = (pred_mask * label).sum().item()
        tn = ((1 - pred_mask) * (1 - label)).sum().item()
        fp = (pred_mask * (1 - label)).sum().item()
        fn = ((1 - pred_mask) * label).sum().item()
        TP += tp
        TN += tn
        FP += fp
        FN += fn
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    print("-" * 30)
    print(f"FINAL TTA RESULTS (Epoch 60):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_tta()
