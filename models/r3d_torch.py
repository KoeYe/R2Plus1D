import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class R3DClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, backbone=None):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None

        self.module = r3d_18(weights=weights)
        in_features = self.module.fc.in_features
        # Replace the final fully connected layer
        self.module.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            return self.module(x)
        else:
            B, N, C, T, H, W = x.shape
            x = x.view(-1, C, T, H, W)
            return self.module(x).view(B, N, -1).mean(1)