import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18 as r2p1d_18_torch

def apply_rope(x, sin, cos):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]          # split channels
    # Expand sin/cos to (1, T, C/2) so they broadcast over batch
    sin, cos = sin.unsqueeze(0), cos.unsqueeze(0)
    rot = torch.cat([x1 * cos - x2 * sin,          # even channels
                     x1 * sin + x2 * cos], dim=-1) # odd  channels
    return rot

class TemporalSelfAttention(nn.Module):
    def __init__(self,
                 inout_channels: int,
                 window_size: int = 3,
                 num_heads: int = 4,
                 reduction_ratio: int = 16,
                 dropout: float = 0.):
        super().__init__()
        self.ws = window_size
        inner_dim = inout_channels // reduction_ratio * 2   # keep it even
        self.patch_embed = nn.Conv3d(inout_channels,
                                     inner_dim,
                                     kernel_size=(1, window_size, window_size),
                                     stride=(1, window_size, window_size),
                                     padding=(0, 0, 0),
                                     bias=False)           # (B, C', T, H/ws, W/ws)

        self.attn = nn.MultiheadAttention(inner_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)

        self.patch_expand = nn.ConvTranspose3d(inner_dim,
                                               inout_channels,
                                               kernel_size=(1, window_size, window_size),
                                               stride=(1, window_size, window_size),
                                               padding=(0, 0, 0),
                                               bias=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(inout_channels)

        # Rotary positional enc.
        self.register_buffer('rope_sin', None, persistent=False)
        self.register_buffer('rope_cos', None, persistent=False)
        self.inner_dim = inner_dim

    # --- RoPE cache ---------------------------------------------------------
    def _build_rope_cache(self, T, device):
        half = self.inner_dim // 2
        freq_seq = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = 1. / (10000 ** (freq_seq / half))
        t = torch.arange(T, device=device).unsqueeze(-1)
        angles = t * inv_freq
        self.rope_sin = angles.sin()      # (T, C/2)
        self.rope_cos = angles.cos()

    def _apply_rope(self, x, sin, cos):
        half = x.size(-1) // 2
        x1, x2 = x[..., :half], x[..., half:]
        sin, cos = sin.unsqueeze(0), cos.unsqueeze(0)
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)
    # ------------------------------------------------------------------------

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        returns: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape
        # spatial window
        x_patch = self.patch_embed(x)        # (B, Cʹ, T, Hʹ, Wʹ)
        # print(x_patch.shape)
        B, Cp, T, Hp, Wp = x_patch.shape
        x_seq = x_patch.permute(0, 3, 4, 2, 1).reshape(B*Hp*Wp, T, Cp)   # (B·Hʹ·Wʹ, T, Cʹ)

        # RoPE position embedding
        # if self.rope_sin is None or self.rope_sin.size(0) < T:
        #     self._build_rope_cache(T, x.device)
        # q = self._apply_rope(x_seq, self.rope_sin[:T], self.rope_cos[:T])
        # k = self._apply_rope(x_seq, self.rope_sin[:T], self.rope_cos[:T])
        # v = x_seq                                           # no RoPE

        # no RoPE
        q = x_seq
        k = x_seq
        v = x_seq

        # causal attention mask
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                            device=x.device), diagonal=1)
        attn_out, _ = self.attn(q, k, v, attn_mask=causal_mask)
        attn_out = attn_out + x_seq          # residual (inner dim)
        attn_out = self.dropout(attn_out)

        # up scale back
        attn_out = attn_out.reshape(B, Hp, Wp, T, Cp) \
                             .permute(0, 4, 3, 1, 2)        # (B, Cʹ, T, Hʹ, Wʹ)
        x_up = self.patch_expand(attn_out)                   # (B, C, T, H, W)
        # print(x_up.shape)
        # ensure shape matching by interpolation
        x_up = F.interpolate(x_up, size=(T, H, W), mode='nearest')

        # final residual + layer norm
        # print(x.shape)
        # print(x_up.shape)
        x = x + x_up # (B, C, T, H, W)
        x = torch.permute(x, (0, 2, 3, 4, 1)) # (B, T, H, W, C)
        out = self.norm(x)
        out = torch.permute(out, (0, 4, 1, 2, 3))
        return out



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
            # TemporalSelfAttention(mid_channels),
            # nn.ReLU(inplace=True), # the output of SA is already normalized
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
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 stride=1,
                 reapply_mid=False,
                 use_mha: bool = False,          # << NEW
                 attn_heads: int = 8):
        super().__init__()

        if mid_channels is None:
            mid_channels = (in_channels * out_channels * 3 * 3 * 3) // (
                in_channels * 3 * 3 + 3 * out_channels)

        # Block 1 (spatial+temporal factorised conv)
        self.conv1 = nn.Sequential(
            Conv2Plus1D(in_channels, out_channels, mid_channels, stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            TemporalSelfAttention(out_channels),
            nn.ReLU(inplace=True), # the output of SA is already normalized
        )

        # Block 2  (temporal enhancement)
        if reapply_mid:
            mid_channels = (out_channels * out_channels * 3 * 3 * 3) // (
                out_channels * 3 * 3 + 3 * out_channels)
        self.conv2 = nn.Sequential(
            Conv2Plus1D(out_channels, out_channels, mid_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            TemporalSelfAttention(out_channels),
        )

        # Skip connection (unchanged)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1,
                          stride=(stride, stride, stride),
                          bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.relu(x + residual)


class R2Plus1D(nn.Module):
    def __init__(self, block, layers, num_classes=400, in_channels=3, use_mha=True, attn_heads=8):
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
        self.layer1 = self._make_layers(block, 64, layers[0], init_stride=1, use_mha=use_mha, attn_heads=attn_heads)
        # print(f"layer2")
        self.layer2 = self._make_layers(block, 128, layers[1], init_stride=2, use_mha=use_mha, attn_heads=attn_heads)
        # print(f"layer3")
        self.layer3 = self._make_layers(block, 256, layers[2], init_stride=2, use_mha=use_mha, attn_heads=attn_heads)
        # print(f"layer4")
        self.layer4 = self._make_layers(block, 512, layers[3], init_stride=2, use_mha=use_mha, attn_heads=attn_heads)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, block, planes, blocks, init_stride, use_mha, attn_heads):
        layers = []
        reapply_mid = blocks > 2
        layers.append(block(self.in_planes, planes,
                            stride=init_stride,
                            reapply_mid=reapply_mid,
                            use_mha=use_mha,
                            attn_heads=attn_heads))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes,
                                reapply_mid=reapply_mid,
                                use_mha=use_mha,
                                attn_heads=attn_heads))
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
        if len(x.shape) == 5:
            return self.r2plus1d(x)
        else:
            B, N, C, T, H, W = x.shape
            x = x.view(-1, C, T, H, W)
            return self.r2plus1d(x).view(B, N, -1).mean(1)

def r2plus1d_18(num_classes: int = 400, in_channels: int = 3, pretrained=False) -> R2Plus1D:
    r2plus1d = R2Plus1D(
        R2Plus1DBlock, [2, 2, 2, 2],
        num_classes=num_classes, in_channels=in_channels
    )
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
        r2plus1d_torch_dict = torch.load("../pretrained_wgts/r2plus1d_34_IG65M.pth")
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    input_tensor = torch.randn(2, 3, 16, 112, 112)

    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)