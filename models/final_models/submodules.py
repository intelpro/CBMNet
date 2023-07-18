import torch
import torch.nn.functional as F
import torch.nn as nn

def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError

class ResBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, in_chs, activation='relu', batch_norm=False):
        super(ResBlock, self).__init__()
        op = []
        for i in range(2):
            op.append(conv3x3(in_chs, in_chs))
            if batch_norm:
                op.append(nn.BatchNorm2d(in_chs))
            if i == 0:
               op.append(actFunc(activation))
        self.main_branch = nn.Sequential(*op)

    def forward(self, x):
        out = self.main_branch(x)
        out += x
        return out

# conv blocks 
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)

def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)

def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, output_padding=1)

# conv resblock
def conv_resblock_three(in_channels, out_channels, stride=1):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels), ResBlock(out_channels))

def conv_resblock_two(in_channels, out_channels, stride=1): 
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels))

def conv_resblock_one(in_channels, out_channels, stride=1):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels))

def conv_1x1_resblock_one(in_channels, out_channels, stride=1):
    return nn.Sequential(conv1x1(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels))

def conv_resblock_two_DS(in_channels, out_channels, stride=2):
    return nn.Sequential(conv3x3(in_channels, out_channels, stride), nn.ReLU(), ResBlock(out_channels), ResBlock(out_channels))

def conv3x3_leaky_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True), nn.LeakyReLU(0.1))

def conv1x1_leaky_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True), nn.LeakyReLU(0.1))