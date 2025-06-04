import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18 as r2p1d_18_torch
# from torchvision.models.video import r2plus1d_34
import os

class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            # spatial
            nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            # Now, we have (B, C, T, H, W)
            # temporal
            nn.Conv3d(
                mid_channels, out_channels, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride: int):
        return stride, stride, stride

class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1, reapply_mid=False):
        super().__init__()

        if mid_channels is None:
            # print("in_channels", in_channels)
            # print("out_channels", out_channels)
            mid_channels = (in_channels * out_channels * 3 * 3 * 3) // (in_channels * 3 * 3 + 3 * out_channels)
            # print("mid_channels: ", mid_channels)

        # convolution block1
        self.conv1 = nn.Sequential(*[
            Conv2Plus1D(in_channels, out_channels, mid_channels, stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ])
        # temporal_conv
        if reapply_mid:
            mid_channels = (out_channels * out_channels * 3 * 3 * 3) // (out_channels * 3 * 3 + 3 * out_channels)
        self.conv2 = nn.Sequential(*[
            Conv2Plus1D(out_channels, out_channels, mid_channels, 1),
            nn.BatchNorm3d(out_channels),
        ])
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(*[
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
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        return F.relu(x)

class R2Plus1D(nn.Module):
    def __init__(self, block, layers, num_classes=400, in_channels=3):
        super().__init__()
        self.in_planes = 64
        # this is what pytorch did
        self.stem = nn.Sequential(*[
            nn.Conv3d(
                in_channels, 45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45, 64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        ])
        # print(f"layer1")
        self.layer1 = self._make_layers(block, 64, layers[0], init_stride=1)
        # print(f"layer2")
        self.layer2 = self._make_layers(block, 128, layers[1], init_stride=2)
        # print(f"layer3")
        self.layer3 = self._make_layers(block, 256, layers[2], init_stride=2)
        # print(f"layer4")
        self.layer4 = self._make_layers(block, 512, layers[3], init_stride=2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, block, planes, blocks, init_stride):
        layers = []
        reapply_mid = True if blocks > 2 else False
        layers.append(block(self.in_planes, planes, stride=init_stride, reapply_mid=reapply_mid))
        self.in_planes = planes

        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, reapply_mid=reapply_mid))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class R2Plus1DClassifier(nn.Module):
    def __init__(self, num_classes, backbone="18", pretrained=True):
        super().__init__()
        if backbone == "18":
            self.r2plus1d = r2plus1d_18(num_classes=num_classes, in_channels=3, pretrained=pretrained)
        elif backbone == "34":
            self.r2plus1d = r2plus1d_34(num_classes=num_classes, in_channels=3, pretrained=pretrained)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.r2plus1d(x)

def r2plus1d_18(num_classes: int = 400, in_channels: int = 3, pretrained=False) -> R2Plus1D:
    r2plus1d = R2Plus1D(R2Plus1DBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        r2plus1d_torch_dict = r2p1d_18_torch(pretrained=pretrained).state_dict()
        r2plus1d_dict = r2plus1d.state_dict()
        filtered_dict = {}

        for k, v in r2plus1d_torch_dict.items():
            # if k.startswith("stem."):
            #     k = k.replace("stem.", "conv1.")
            # if k.startswith("layer"):
            #     k = k.replace("conv1", "spatial_conv")
            #     k = k.replace("conv2", "temporal_conv")
            if k in r2plus1d_dict and v.size() == r2plus1d_dict[k].size():
                print(f"{k} is matching!")
                filtered_dict[k] = v
            elif k in r2plus1d_dict:
                print(f"{k} s shape is NOT matching! we got {v.size()} but expected {r2plus1d_dict[k].size()}")
            else:
                print(f"{k} is not expected")

        missing, unexpected = r2plus1d.load_state_dict(filtered_dict, strict=False)
        # print(f"Expected Keys: {r2plus1d_dict.keys()}")
        # print(f"Loaded Keys: {set(filtered_dict.keys()) & set(r2plus1d_dict.keys())}")
        # print(f"Missing Keys: {missing}")
        # print(f"Unexpected Keys: {unexpected}")
    return r2plus1d

# no pretrained model for r2plus1d_34 in pytorch
def r2plus1d_34(num_classes: int = 400, in_channels: int = 3, pretrained=False) -> R2Plus1D:
    r2plus1d = R2Plus1D(R2Plus1DBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        r2plus1d_torch_dict = torch.load(os.path.join(os.path.dirname(__file__), "..", "r2plus1d_34_IG65M.pth"))
        r2plus1d_dict = r2plus1d.state_dict()

        filtered_dict = {}
        for k, v in r2plus1d_torch_dict.items():
            # if k.startswith("stem."):
            #     k = k.replace("stem.", "conv1.")
            # if k.startswith("layer"):
            #     k = k.replace("conv1", "spatial_conv")
            #     k = k.replace("conv2", "temporal_conv")
            if k in r2plus1d_dict and v.size() == r2plus1d_dict[k].size():
                # print(f"{k} is matching!")
                filtered_dict[k] = v
            # elif k in r2plus1d_dict:
            #     print(f"{k} s shape is NOT matching! we got {v.size()} but expected {r2plus1d_dict[k].size()}")
            # else:
            #     print(f"{k} is not expected")


        missing, unexpected = r2plus1d.load_state_dict(filtered_dict, strict=False)
        # print(f"Expected Keys: {r2plus1d_dict.keys()}")
        # print(f"Loaded Keys: {set(filtered_dict.keys()) & set(r2plus1d_dict.keys())}")
        # print(f"Missing Keys: {missing}")
        # print(f"Unexpected Keys: {unexpected}")
    return r2plus1d

if __name__ == '__main__':
    model = r2plus1d_18(num_classes=10, in_channels=3, pretrained=True)
    model.eval()

    input_tensor = torch.randn(2, 3, 16, 112, 112)

    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)