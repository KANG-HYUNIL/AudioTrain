import torch
import torch.nn as nn
from models import register_student

class MultiBranchDepthwiseConv(nn.Module):
    """
    Depthwise convolution block with 3x3, 1x1, 3x1, 1x3 branches.
    After training, branches can be reparameterized and merged.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1, padding=0, groups=in_channels)
        self.conv3x1 = nn.Conv2d(in_channels, in_channels, (3,1), padding=(1,0), groups=in_channels)
        self.conv1x3 = nn.Conv2d(in_channels, in_channels, (1,3), padding=(0,1), groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.reparam = False
        self.merged_conv = None

    def forward(self, x):
        if self.reparam and self.merged_conv is not None:
            out = self.merged_conv(x)
        else:
            out = self.conv3x3(x) + self.conv1x1(x) + self.conv3x1(x) + self.conv1x3(x)
        out = self.bn(out)
        return self.act(out)

    def reparameterize(self):
        """
        Merge all branch kernels into a single 3x3 kernel for inference.
        """
        with torch.no_grad():
            # Expand 1x1 kernel to 3x3
            k1x1 = torch.zeros_like(self.conv3x3.weight)
            k1x1[:,:,1,1] = self.conv1x1.weight.squeeze(-1).squeeze(-1)
            # Expand 3x1 kernel to 3x3
            k3x1 = torch.zeros_like(self.conv3x3.weight)
            k3x1[:,:, :,1] = self.conv3x1.weight.squeeze(-1)
            # Expand 1x3 kernel to 3x3
            k1x3 = torch.zeros_like(self.conv3x3.weight)
            k1x3[:,:,1,:] = self.conv1x3.weight.squeeze(-2)
            # Sum all kernels
            merged_weight = self.conv3x3.weight + k1x1 + k3x1 + k1x3
            # Bias 병합
            merged_bias = None
            if self.conv3x3.bias is not None:
                merged_bias = self.conv3x3.bias.clone()
                for b in [self.conv1x1.bias, self.conv3x1.bias, self.conv1x3.bias]:
                    if b is not None:
                        merged_bias += b
            # 새로운 depthwise conv 생성
            self.merged_conv = nn.Conv2d(
                self.conv3x3.in_channels,
                self.conv3x3.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.conv3x3.in_channels,
                bias=merged_bias is not None
            )
            self.merged_conv.weight.data.copy_(merged_weight)
            if merged_bias is not None:
                self.merged_conv.bias.data.copy_(merged_bias)
            else:
                self.merged_conv.bias = None
            self.reparam = True

class StudentMultiBranchNet(nn.Module):
    """
    Custom student model with multi-branch depthwise conv blocks.
    """
    def __init__(self, num_classes=11, in_channels=1, width_mult=0.75):
        super().__init__()
        ch = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU()
        )
        self.block1 = MultiBranchDepthwiseConv(ch)
        self.block2 = MultiBranchDepthwiseConv(ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def reparameterize(self):
        self.block1.reparameterize()
        self.block2.reparameterize()

@register_student("multibranch")
def build_student_multibranch(num_classes: int = 11, in_channels: int = 1, width_mult: float = 0.75):
    return StudentMultiBranchNet(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult)
