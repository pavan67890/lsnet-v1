import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from models.lsnet import LSNet
from losses.bce_dice import BCEDiceLoss
from utils import count_params, set_seed
import os

def main():
    set_seed(42)
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    B = 8
    H = 256
    W = 256
    model = LSNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = GradScaler()
    enc_params = count_params(model.encoder)
    total_params = count_params(model)
    torch.cuda.reset_peak_memory_stats(device)
    out_shape = None
    logs = []
    with open("s:\\sota\\checks_result.txt", "w") as f:
        f.write("start\n")
    for i in range(3):
        t1 = torch.randn(B, 3, H, W, device=device)
        t2 = torch.randn(B, 3, H, W, device=device)
        m = torch.rand(B, 1, H, W, device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits, probs = model(t1, t2)
            out_shape = tuple(logits.shape)
            loss = criterion(logits, m)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("loss", float(loss.item()))
        logs.append("loss " + str(float(loss.item())))
    max_mem = torch.cuda.max_memory_allocated(device)
    gb = max_mem / (1024 ** 3)
    print("encoder_params", enc_params)
    print("total_params", total_params)
    print("output_shape", out_shape)
    print("max_mem_gb", gb)
    logs.append("encoder_params " + str(enc_params))
    logs.append("total_params " + str(total_params))
    logs.append("output_shape " + str(out_shape))
    logs.append("max_mem_gb " + str(gb))
    with open("s:\\sota\\checks_result.txt", "w") as f:
        f.write("\n".join(logs))

if __name__ == "__main__":
    main()
