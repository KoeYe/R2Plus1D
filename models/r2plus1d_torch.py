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
        return self.r2plus1d(x)
