import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from video_datasets import HuggingFaceSSV2Dataset
# from models.r2plus1d import R2Plus1DClassifier
# from models.r2plus1d_attn import R2Plus1DClassifier
# from models.r2plus1d_attn_v3 import R2Plus1DClassifier
from models.r2plus1d_attn_v4 import R2Plus1DClassifier
# from models.r2plus1d_torch import R2Plus1DClassifier
# from torchvision.models.video import r3d_18
from models.tsm import tsm_res50
from models.r3d_torch import R3DClassifier

import tqdm
from torch.utils.data import DataLoader
from trainer import Trainer
import matplotlib.pyplot as plt
import multiprocessing
import os
from torch.amp import autocast, GradScaler

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root ="./data/something-something-v2"
num_clips = 2

# model = R2Plus1DClassifier(num_classes=174, backbone="18")
# model.load_state_dict(torch.load("output/r2plus1d_18_attnv4_best.pt"))

model = R3DClassifier(num_classes=174).to(device)
state_dict = torch.load("output/r3d.pt")
# state_dict = {'.'.join(key.split('.')[1:]): state_dict[key] for key in state_dict.keys()}
# print(state_dict.keys())
model.load_state_dict(state_dict)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

val_set = HuggingFaceSSV2Dataset(data_root, data_split='validation', temporal_random=True, num_clip_eval=num_clips)
val_loader = DataLoader(val_set, batch_size=64//num_clips, shuffle=False, num_workers=14)


# model = tsm_res50(path='./pretrained_wgts/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth')
# val_set = HuggingFaceSSV2Dataset(data_root, data_split='validation', temporal_random=True, two_clip=True, use_standard_labels=True, num_clip_eval=num_clips)
# val_loader = DataLoader(val_set, batch_size=64//num_clips, shuffle=False, num_workers=14)

# 使用多GPU训练
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)  # 将模型包装为DataParallel

model = model.to(device)

model.eval()
total_loss, correct, total = 0.0, 0, 0
criterion = nn.CrossEntropyLoss()
pbar = tqdm.tqdm(val_loader, desc=f"Testing", unit="batch", leave=False)
with torch.no_grad():
    for videos, labels in pbar:
        videos = videos.to(device)
        labels = labels.to(device)

        with autocast(device_type='cuda'):  # Enable mixed precision evaluation
            logits = model(videos)
            loss = criterion(logits, labels)

        total_loss += loss.item() * videos.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += videos.size(0)

        running_loss = total_loss / total
        running_acc = correct / total
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

avg_loss = total_loss / total
acc = correct / total

print(f"Total loss: {avg_loss:.4f}, acc: {acc:.4f}")