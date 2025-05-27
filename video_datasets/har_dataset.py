import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from pathlib import Path
import random
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize

class HARVideoDataset(Dataset):
    def __init__(self, root_dir, data_split='train', clip_len=8, frame_step=1, transform=None):
        self.samples = []
        self.transform = transform
        root = Path(root_dir)
        # assume structure: root/<label>/*.mp4
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir(): continue
            label = class_dir.name
            for vid in sorted(class_dir.glob("*.mp4")):
                self.samples.append((vid, label))
        if data_split == 'train':
            self.samples = self.samples[:100]
        elif data_split == 'validation':
            self.samples = self.samples[100:120]
        else:
            self.samples = self.samples[120:]
        # build a label→index map
        labels = sorted({lbl for _, lbl in self.samples})
        self.cls2idx = {lbl:i for i,lbl in enumerate(labels)}
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
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # read_video returns (Tensor[T,H,W,C], Tensor[T_audio], info)
        video, _, _ = read_video(str(path), pts_unit="sec")

        total_frames = video.shape[0]
        if self.clip_len > 0:
            max_start = max(0, total_frames - self.clip_len * self.frame_step)
            start_idx = random.randint(0, max_start) if hasattr(self, 'mode') and self.mode == 'train' else 0

            indices = [min(start_idx + i * self.frame_step, total_frames - 1) for i in range(self.clip_len)]
            clip = video[indices]  # (clip_len, H, W, 3)
        else:
            clip = video  # (T, H, W, 3)

        clip = clip / 255.0
        if self.transform:
            # Apply per-frame
            clip = torch.stack([self.transform(frame) for frame in clip])

        clip = clip.permute(1, 0, 2, 3).contiguous()

        return clip, self.cls2idx[label]

if __name__ == '__main__':
    dataset = HARVideoDataset('../data/HAR')
    print(len(dataset))
    video, label_idx = dataset[0]
    print(video.shape, label_idx)
