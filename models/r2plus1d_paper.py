import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18 as r2p1d_18_torch
# from torchvision.models.video import r2plus1d_34

class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.spatial_conv = nn.Sequential(*[
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        ])
        self.temporal_conv = nn.Sequential(*[
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(1, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
        ])
        self.skip_conn = None
        if stride != 1 or in_channels != out_channels:
            self.skip_conn = nn.Sequential(*[
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, stride, stride),
                    bias=False
                ),
                nn.BatchNorm3d(out_channels),
            ])

    def forward(self, x):
        residual = x
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        if self.skip_conn is not None:
            residual = self.skip_conn(residual)
        x += residual
        return F.relu(x)

class R2Plus1D(nn.Module):
    def __init__(self, block, layers, num_classes=400, in_channels=3):
        super().__init__()
        self.in_planes = 64
        # this is what paper did
        self.conv1 = nn.Sequential(*[
            nn.Conv3d(
                in_channels, self.in_planes,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1)
            )
        ])
        self.layer1 = self._make_layers(block, 64, layers[0], init_stride=1)
        self.layer2 = self._make_layers(block, 128, layers[1], init_stride=2)
        self.layer3 = self._make_layers(block, 256, layers[2], init_stride=2)
        self.layer4 = self._make_layers(block, 512, layers[3], init_stride=2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, block, planes, blocks, init_stride):
        strides = [init_stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def r2plus1d_18(num_classes: int = 400, in_channels: int = 3) -> R2Plus1D:
    r2plus1d = R2Plus1D(R2Plus1DBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


    return r2plus1d

# no pretrained model for r2plus1d_34 in pytorch
def r2plus1d_34(num_classes: int = 400, in_channels: int = 3, pretrained=False) -> R2Plus1D:
    r2plus1d = R2Plus1D(R2Plus1DBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        r2plus1d_torch_dict = torch.load("r2plus1d_34_IG65M.pth")
        r2plus1d_dict = r2plus1d.state_dict()

        filtered_dict = {}
        for k, v in r2plus1d_torch_dict.items():
            if k.startswith("stem."):
                k = k.replace("stem.", "conv1.")
            if k.startswith("layer"):
                k = k.replace("conv1", "spatial_conv")
                k = k.replace("conv2", "temporal_conv")
            if k in r2plus1d_dict and v.size() == r2plus1d_dict[k].size():
                print(f"{k} is matching!")
                filtered_dict[k] = v
            elif k in r2plus1d_dict:
                print(f"{k} s shape is NOT matching! we got {v.size()} but expected {r2plus1d_dict[k].size()}")
            else:
                print(f"{k} is not expected")


        missing, unexpected = r2plus1d.load_state_dict(filtered_dict, strict=False)
        print(f"Expected Keys: {r2plus1d_dict.keys()}")
        print(f"Loaded Keys: {set(filtered_dict.keys()) & set(r2plus1d_dict.keys())}")
        print(f"Missing Keys: {missing}")
        print(f"Unexpected Keys: {unexpected}")
    return r2plus1d

if __name__ == '__main__':
    model = r2plus1d_34(num_classes=10, in_channels=3, pretrained=True)
    model.eval()

    input_tensor = torch.randn(2, 3, 16, 112, 112)

    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)