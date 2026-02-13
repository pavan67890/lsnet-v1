DATA_ROOT = "S:/sota/levir_project/datasets"
PATCHES_ROOT = "S:/sota/levir_project/levir_patches"
CHECKPOINTS_DIR = "S:/sota/levir_project/checkpoints"
LOGS_DIR = "S:/sota/levir_project/logs"
OUTPUTS_DIR = "S:/sota/levir_project/outputs"

patch_size = 256
train_stride = 200
val_stride = 256
test_stride = 256

micro_batch_size = 4
grad_accum_steps = 2
epochs = 100

lr = 5e-4
weight_decay = 1e-2
num_workers = 2

amp = True
cudnn_benchmark = True
seed = 42
