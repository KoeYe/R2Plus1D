import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, backbone=None):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None

        self.r2plus1d = r2plus1d_18(weights=weights)
        in_features = self.r2plus1d.fc.in_features
        # Replace the final fully connected layer
        self.r2plus1d.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            return self.r2plus1d(x)
        else:
            B, N, C, T, H, W = x.shape
            x = x.view(-1, C, T, H, W)
            return self.r2plus1d(x).view(B, N, -1).mean(1)