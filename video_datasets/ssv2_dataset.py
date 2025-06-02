import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize

# For raw video decoding
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None
    # Decord is optional if using Hugging Face video decoding

# For Hugging Face dataset wrapper
from datasets import load_dataset

class HuggingFaceSSV2Dataset(Dataset):
    def __init__(self, data_dir, data_split='train', clip_len=8, frame_step=1, transform=None):
        self.video_dir = os.path.join(data_dir, "20bn-something-something-v2")
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        # Load and decode videos via Hugging Face dataset library
        print(f"loading {data_split} dataset files, it may take a while...")
        self.hf_dataset = load_dataset(
            data_dir,
            split=data_split,
        )
        print(f"{data_split} dataset loaded")

        self.idx2templates = sorted(set(self.hf_dataset["template"]))
        self.template2idx = {template: idx for idx, template in enumerate(self.idx2templates)}
        self.clip_len = clip_len
        self.frame_step = frame_step
        self.transform = transform
        if self.transform is None:
            mean = [0.43216, 0.394666, 0.37645]
            std = [0.22803, 0.22145, 0.216989]
            self.transform = Compose([
                ToPILImage(),
                Resize((128, 171)),
                CenterCrop((112, 112)),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        id = example['id']
        filename = os.path.join(self.video_dir, f"{id}.webm")
        vr = VideoReader(filename, ctx=cpu(0), num_threads=1)

        video = vr.get_batch(range(len(vr))).asnumpy() # (T, H, W, 3)

        label = example.get("template")
        label = self.template2idx[label]

        total_frames = video.shape[0]
        frame_step = total_frames // self.clip_len
        if self.clip_len > 0:
            max_start = max(0, total_frames - self.clip_len * frame_step)
            start_idx = random.randint(0, max_start) if hasattr(self, 'mode') and self.mode == 'train' else 0

            indices = [min(start_idx + i * frame_step, total_frames - 1) for i in range(self.clip_len)]
            clip = video[indices]  # (clip_len, H, W, 3)
        else:
            clip = video  # (T, H, W, 3)

        clip = clip / 255.0
        if self.transform:
            # Apply per-frame
            clip = torch.stack([self.transform(frame) for frame in clip])

        clip = clip.permute(1, 0, 2, 3).contiguous()

        return clip, label

if __name__ == '__main__':
    dataset = HuggingFaceSSV2Dataset("../data/something-something-v2", data_split='train')
    print(len(dataset))
    video, label = dataset[0]
    print(video.shape)
    print(label)