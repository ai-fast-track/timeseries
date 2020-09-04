import torch
import torch.nn as nn
from fastai.torch_core import *
from fastai.layers import *
from fastai.imports import *

# Iception Time paper: https://arxiv.org/abs/1909.04939

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


act_fn = nn.ReLU(inplace=True)
def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

class Shortcut(Module):
    "Merge a shortcut with the result of the module by adding them. Adds Conv, BN and ReLU"
    def __init__(self, ni, nf, act_fn=act_fn): 
        self.act_fn=act_fn
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)
    def forward(self, x): return act_fn(x + self.bn(self.conv(x.orig)))

class InceptionModule(Module):
    def __init__(self, ni, nb_filters=32, kss=[39, 19, 9], bottleneck_size=32, stride=1):
        self.bottleneck = nn.Conv1d(ni, bottleneck_size, 1) if (bottleneck_size>0 and ni>1) else noop
        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>1 and ni>1) else ni, nb_filters, ks) for ks in kss])
        self.maxpool_bottleneck = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())
    def forward(self, x):
        bottled = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.maxpool_bottleneck(x)], dim=1))

def inception_time(ni, nout, ks=40, depth=6, bottleneck_size=32, nb_filters=32, head=True):
    layers = []
    
    # compute kernel sizes: eg for ks=40 => kss=[39, 19, 9] 
    kss = [ks // (2**i) for i in range(3)]
    # ensure odd kss until nn.Conv1d with padding='same' is available in pytorch 1.3
    kss = [ksi if ksi % 2 != 0 else ksi - 1 for ksi in kss]
    n_ks = len(kss) + 1
    for d in range(depth):
        # Farid
      # im = SequentialEx(InceptionModule(ni if d==0 else n_ks*nb_filters, kss=kss, bottleneck_size=bottleneck_size))
        im = SequentialEx(InceptionModule(ni if d==0 else n_ks*nb_filters, kss=kss, bottleneck_size=bottleneck_size if d > 0 else 0))
        if d%3==2: im.append(Shortcut(n_ks*nb_filters, n_ks*nb_filters))      
        layers.append(im)
    head = [AdaptiveConcatPool1d(), Flatten(), nn.Linear(2*n_ks*nb_filters, nout)] if head else []
    return  nn.Sequential(*layers, *head)
