import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ----------------------------
# 1. Temporal Shift Module
# ----------------------------
class TemporalShift(nn.Module):
    """Temporal Shift Module (TSM). Shifts part of the channels along the temporal dim.
    See: Ji Lin et al. ICCV 2019 ."""
    def __init__(self, n_segment=8, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        # x: [N*T, C, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        # shift left
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        # no shift
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        return out.view(nt, c, h, w)

class TSM(nn.Module):
    """Wrap a ResNet backbone with TSM inserted before each conv1."""
    def __init__(self, num_classes=174, num_segments=8, backbone='resnet50',
                 pretrained_base=True, shift_div=8, shift_place='blockres'):
        super().__init__()
        self.num_segments = num_segments
        # load a ResNet
        self.base_model = getattr(models, backbone)(pretrained=pretrained_base)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        # insert TSM into every residual block's first conv
        from torchvision.models.resnet import Bottleneck, BasicBlock
        for m in self.base_model.modules():
            if isinstance(m, (Bottleneck, BasicBlock)):
                m.conv1 = nn.Sequential(
                    TemporalShift(n_segment=num_segments, n_div=shift_div),
                    m.conv1
                )

    def forward(self, x):
        if len(x.shape) == 5:
            # x in: [B, C, T, H, W]
            # x: [B, T, C, H, W]
            x = torch.permute(x, (0, 2, 1, 3, 4)).contiguous()
            b, t, c, h, w = x.size()
            # print(x.size())
            x = x.view(b * t, c, h, w)
            y = self.base_model(x)             # [B*T, num_classes]
            y = y.view(b, t, -1).mean(1)       # segment consensus by average
            # print(y.shape)
            return y
        else:
            B, N, C, T, H, W = x.shape
            # print(x.shape)
            x = x.view(-1, C, T, H, W)
            x = torch.permute(x, (0, 2, 1, 3, 4)).contiguous()
            b, t, c, h, w = x.size()
            # print(x.size())
            x = x.view(b * t, c, h, w)
            y = self.base_model(x)  # [B*T, num_classes]
            y = y.view(b, t, -1).mean(1)  # segment consensus by average
            # print(y.shape)
            y = y.view(B, N, -1).mean(1)
            # print(y.shape)
            return y


from collections import OrderedDict
import re
def load_tsm_stm_ckpt(model, ckpt_path):
    """
    Load an SSV2 checkpoint into the TSM/STM model that:
      • was saved from DataParallel (prefix 'module.')
      • stores conv1 weight as '*.conv1.net.*'
      • stores head as 'new_fc.*'
    """
    raw = torch.load(ckpt_path, map_location='cpu')
    sd  = raw['state_dict'] if 'state_dict' in raw else raw

    new_sd = OrderedDict()
    for k, v in sd.items():
        # 1) drop leading 'module.' if present
        if k.startswith('module.'):
            k = k[7:]

        # 2) conv1.net → conv1.1   (only once per key)
        k = re.sub(r'\.conv1\.net\.', '.conv1.1.', k, count=1)

        # 3) new_fc → base_model.fc
        if k.startswith('new_fc.'):
            k = 'base_model.' + k.replace('new_fc.', 'fc.', 1)

        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("✅ loaded with renamed keys.")
    if missing:
        print("   ⦻ still missing:", missing[:5], "…" if len(missing) > 5 else "")
    if unexpected:
        print("   ✔ ignored:", unexpected[:5], "…" if len(unexpected) > 5 else "")

def tsm_res50(path):
    tsm = TSM(num_classes=174, num_segments=8)
    load_tsm_stm_ckpt(tsm, path)
    return tsm

if __name__ == '__main__':
    # 3.1 TSM pretrained on SSV2
    tsm = TSM(num_classes=174, num_segments=8)
    load_tsm_stm_ckpt(tsm, '../pretrained_wgts/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth')

    total_params = sum(p.numel() for p in tsm.parameters())
    print(f"Total parameters: {total_params}")