import os
import json
import random
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToPILImage, Resize, CenterCrop, ToTensor, Normalize, RandomCrop

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
    def __init__(self, data_dir, data_split='train', clip_len=8, transform=None, temporal_random=True):
        self.video_dir = os.path.join(data_dir, "20bn-something-something-v2")
        if not os.path.isdir(self.video_dir):
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")

        # Load and decode videos via Hugging Face dataset library
        print(f"loading {data_split} dataset files, it may take a while...")
        # self.hf_dataset = load_dataset(
        #     data_dir,
        #     split=data_split,
        #     # streaming=True,
        #     num_proc=32
        # )
        with open(os.path.join(data_dir, "labels", f"{data_split}.json"), 'r') as f:
            self.data = json.load(f)
        # print(data[:10])
        print(f"{data_split} dataset loaded")

        self.mode = data_split
        self.temporal_random = temporal_random
        self.idx2templates = sorted(set(self.data[i]["template"] for i in range(len(self.data))))
        self.template2idx = {template: idx for idx, template in enumerate(self.idx2templates)}
        self.clip_len = clip_len
        self.transform = transform
        if self.transform is None:
            mean = [0.43216, 0.394666, 0.37645]
            std = [0.22803, 0.22145, 0.216989]
            # self.transform = Compose([
            #     ToPILImage(),
            #     Resize((128, 171)),
            #     CenterCrop((112, 112)),
            #     ToTensor(),
            #     Normalize(mean=mean, std=std)
            # ])
            if self.mode == 'train':
                self.transform = Compose([
                ToPILImage(),
                Resize((256)),
                RandomCrop((224)),
                ToTensor(),
                Normalize(mean=mean, std=std)
                ])
            else:
                self.transform = Compose([
                ToPILImage(),
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean, std)
                ])

    def __len__(self):
        return len(self.data)

    @lru_cache(maxsize=16)
    def _get_videoreader(self, filepath):
        return VideoReader(filepath, ctx=cpu(0), num_threads=1)

    def __getitem__(self, idx):
        example = self.data[idx]
        id = example['id']
        filename = os.path.join(self.video_dir, f"{id}.webm")
        # vr = VideoReader(filename, ctx=cpu(0), num_threads=1)
        # print(os.path.exists(filename))
        vr = self._get_videoreader(filename)

        video = vr.get_batch(range(len(vr))).asnumpy() # (T, H, W, 3)

        label = example.get("template")
        label = self.template2idx[label]

        total_frames = video.shape[0]
        if not self.temporal_random:
            frame_step = total_frames // self.clip_len
            if self.clip_len > 0:
                max_start = max(0, total_frames - self.clip_len * frame_step)
                start_idx = random.randint(0, max_start) if hasattr(self, 'mode') and self.mode == 'train' else 0

                indices = [min(start_idx + i * frame_step, total_frames - 1) for i in range(self.clip_len)]
                clip = video[indices]  # (clip_len, H, W, 3)
            else:
                clip = video  # (T, H, W, 3)
        else:
            segment_size = total_frames / float(self.clip_len)
            indices = []
            for i in range(self.clip_len):
                start_f = i * segment_size
                end_f = (i + 1) * segment_size

                seg_start = int(np.floor(start_f))
                seg_end = int(np.floor(end_f))
                if seg_end <= seg_start:
                    seg_end = min(seg_start + 1, total_frames)

                if self.mode == "train":
                    # pick random in [seg_start, seg_end-1]
                    chosen = random.randint(seg_start, seg_end - 1)
                else:
                    # center frame: average of seg_start and seg_end, truncated
                    chosen = (seg_start + seg_end) // 2
                chosen = min(chosen, total_frames - 1)
                indices.append(chosen)
            clip = vr.get_batch(indices).asnumpy()

        clip = clip / 255.0
        if self.transform:
            # Apply per-frame
            clip = torch.stack([self.transform(frame) for frame in clip])

        clip = clip.permute(1, 0, 2, 3).contiguous()

        clip = torch.as_tensor(clip, dtype=torch.float16)

        return clip, label
    
import matplotlib.pyplot as plt
def visualize_clip(clip_tensor, mean, std):
    """
    Given a clip tensor of shape (C, T, H, W) that was normalized
    with Normalize(mean, std), unnormalize it and plot each frame
    in a row of subplots.
    """
    C, T, H, W = clip_tensor.shape
    # Move to CPU / numpy
    clip_np = clip_tensor.cpu().numpy()  # still normalized

    # Unnormalize per‐channel
    for c in range(C):
        clip_np[c, :, :, :] = clip_np[c, :, :, :] * std[c] + mean[c]

    # Now clip_np is in [0,1] (approximately). Rearrange to (T, H, W, C)
    clip_np = np.transpose(clip_np, (1, 2, 3, 0))

    # Plot
    fig, axes = plt.subplots(1, T, figsize=(T * 3, 3))
    if T == 1:
        axes = [axes]  # make it iterable if only one frame

    for t in range(T):
        frame = clip_np[t]
        # Clip to [0,1] just in case of tiny numerical overshoot
        frame = np.clip(frame, 0.0, 1.0)
        axes[t].imshow(frame)
        axes[t].axis("off")
        axes[t].set_title(f"Frame {t}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dataset = HuggingFaceSSV2Dataset("../data/something-something-v2", data_split='train', temporal_random=True)
    samples = 5
    print(len(dataset))
    for _ in range(samples):
        idx = np.random.randint(len(dataset))
        video, label = dataset[idx]
        print(video.shape)
        print(label)
        print(dataset.idx2templates[label])

        # The same mean/std that the dataset’s default transform used:
        mean = [0.43216, 0.394666, 0.37645]
        std = [0.22803, 0.22145, 0.216989]

        visualize_clip(video, mean, std)
