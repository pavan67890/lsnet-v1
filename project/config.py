import os

BASE_DIR = os.path.dirname(__file__)
data_root = os.path.join(BASE_DIR, "data", "whu")
t1_dir = os.path.join(data_root, "t1")
t2_dir = os.path.join(data_root, "t2")
mask_dir = os.path.join(data_root, "mask")

micro_batch_size = 4
grad_accum_steps = 2
effective_batch_size = micro_batch_size * grad_accum_steps
epochs = 100
lr = 5e-4
weight_decay = 1e-4
warmup_epochs = 5
patch_size = 256
train_stride = 128
test_stride = 256
num_workers = 4
seed = 42
