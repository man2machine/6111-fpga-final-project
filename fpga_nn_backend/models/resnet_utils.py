import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_places, out_planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_places, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out

class ResNetBuilder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _initialize_weights(self, initialize_linear=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def _make_layer(self, in_planes, out_planes, num_blocks, stride=1, block=BasicBlock):
        downsample = None
        if stride != 1 or in_planes != out_planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * block.expansion),
            )

        layers = []
        layers.append(block(in_planes, out_planes, stride, downsample))
        in_planes = out_planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(in_planes, out_planes))

        return nn.Sequential(*layers)