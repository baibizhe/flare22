
import torch
import torch.nn as nn
import torch.nn.functional as F

class IBN3d(nn.Module):
    def __init__(self, planes):
        super(IBN3d, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm3d(half1, affine=True)
        self.BN = nn.BatchNorm3d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def norm3d(norm, planes):
    if norm == 'ibn':
        return IBN3d(planes), nn.BatchNorm3d(planes), IBN3d(planes)
    elif norm == 'bn':
        return nn.BatchNorm3d(planes), nn.BatchNorm3d(planes), nn.BatchNorm3d(planes)
    elif norm == 'in':
        return nn.InstanceNorm3d(planes), nn.InstanceNorm3d(planes), nn.InstanceNorm3d(planes)


class BasicBlock3D(nn.Module):
    def __init__(self, inplanes, planes, norm='bn'):
        super(BasicBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes)
        self.bn1, self.bn2, self.bn3 = norm3d(norm, planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResidualBlock3D(nn.Module):
    def __init__(self, inplanes, planes, norm='bn'):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes)
        self.bn1, self.bn2, self.bn3 = norm3d(norm, planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        if inplanes != planes:
            self.scale = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1))
        self.inplanes = inplanes
        self.planes = planes

    def forward(self, x):
        if self.inplanes != self.planes:
            residual = self.scale(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


def Down(planes, kernel_size=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size)


class Up(nn.Module):
    def __init__(self, in_planes, out_planes, norm, scale_factor=(2, 2, 2)):
        super(Up, self).__init__()

        self.scale_factor = scale_factor
        self.conv = BasicBlock3D(in_planes, out_planes, norm=norm)

    def forward(self, inputs1, inputs2):
        inputs2 = F.interpolate(inputs2, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        return self.conv(torch.cat([inputs1, inputs2], 1))


class UNET(nn.Module):
    def __init__(self, norm='bn', feature_scale=2,outputChannel=12):
        super(UNET, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x // feature_scale) for x in filters]
        self.conv1 = BasicBlock3D(1, filters[0], norm)
        self.downsample1 = Down(filters[0])
        self.conv2 = BasicBlock3D(filters[1]//2, filters[1], norm)
        self.downsample2 = Down(filters[1])
        self.conv3 = BasicBlock3D(filters[2]//2, filters[2], norm)
        self.downsample3 = Down(filters[2])
        self.conv4 = BasicBlock3D(filters[3]//2, filters[3], norm)
        self.downsample4 = Down(filters[3])
        self.center = BasicBlock3D(filters[4]//2, filters[4], norm)
        self.up4 = Up(filters[4] + filters[3], filters[3], norm)
        self.up3 = Up(filters[3] + filters[2], filters[2], norm)
        self.up2 = Up(filters[2] + filters[1], filters[1], norm)
        self.up1 = Up(filters[1] + filters[0], filters[0], norm)
        self.final = nn.Conv3d(filters[0], outputChannel, 1)
        self.activate = nn.Sigmoid()
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm3d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.InstanceNorm3d):
#                 init_weights(m, init_type='kaiming')

    #     self.initialize()
    #
    # def initialize(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             nn.init.normal_(m.weight.data, 1.0, 0.02)
    #             nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        down1 = self.downsample1(conv1)
        conv2 = self.conv2(down1)
        down2 = self.downsample2(conv2)
        conv3 = self.conv3(down2)
        down3 = self.downsample3(conv3)
        conv4 = self.conv4(down3)
        up4 = self.up4(conv4, self.center(self.downsample4(conv4)))
        up3 = self.up3(conv3, up4)
        up2 = self.up2(conv2, up3)
        up1 = self.up1(conv1, up2)
#         return self.activate(self.final(up1))
        return self.final(up1)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputChannel = 12
    x = torch.Tensor(2, 1, 64 , 64, 64)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = UNET(outputChannel=outputChannel)

    out = model(x)
    print("out size: {}".format(out.size()))