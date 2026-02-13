import os
import torch
from models.lsnet import LSNet
import train as train_mod

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSNet().to(device)
    cp_path = r"S:\sota\project\checkpoints\lsnet_whu_best.pth"
    if not os.path.exists(cp_path):
        print("Checkpoint not found:", cp_path)
        return
    payload = torch.load(cp_path, map_location=device)
    state = payload.get("model", None)
    if state is None:
        try:
            model.load_state_dict(payload)
        except Exception:
            print("Invalid checkpoint format. Abort.")
            return
    else:
        model.load_state_dict(state)
    print("Loaded checkpoint:", cp_path)
    model.eval()
    train_loader, val_loader = train_mod.build_dataloaders()
    p, r, f1, iou = train_mod.evaluate(model, val_loader, device)
    print("precision", round(p, 4))
    print("recall", round(r, 4))
    print("f1", round(f1, 4))
    print("iou", round(iou, 4))

if __name__ == "__main__":
    main()
