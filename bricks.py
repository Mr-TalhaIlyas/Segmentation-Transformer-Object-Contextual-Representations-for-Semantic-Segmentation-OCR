import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv3x3(nn.Module):
    '''Depth wise conv'''
    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)
        
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels, norm_type=config['norm_typ'])
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class twiceConvBNRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()
        self.conv1 = ConvBNRelu(inChannels, outChannels, kernel=kernel, stride=stride, padding=padding,
                                dilation=dilation, groups=groups)
        self.conv2 = ConvBNRelu(outChannels, outChannels, kernel=kernel, stride=stride, padding=padding,
                                dilation=dilation, groups=groups)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SeprableConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernal_size=3, bias=False):
        self.dwconv = nn.Conv2d(inChannels, inChannels, kernal_size=kernal_size,
                                groups=inChannels, bias=bias)
        self.pwconv = nn.Conv2d(inChannels, inChannels, kernal_size=1, bias=bias)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        
        return x

class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.act(x)
        
        return x

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class AUX_Head(nn.Module):
    '''
    Used for calculating auxiliary loss before final loss during training
    from ground truths.
    '''
    def __init__(self, inChannels, num_classes, kernel=1, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inChannels, inChannels, kernel_size=kernel,
                               padding=0, stride=stride, dilation=dilation,
                               groups=groups, bias=True)
        self.norm = NormLayer(inChannels, norm_type=config['norm_typ'])
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inChannels, num_classes, kernel_size=kernel,
                               padding=0, stride=stride, dilation=dilation,
                               groups=groups, bias=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)

        return x
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
