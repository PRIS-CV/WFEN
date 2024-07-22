from models.arch.blocks import *
import torch
from torch import nn, einsum
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange


class GSA(nn.Module):
    def __init__(self, channels, num_heads=8, bias=False):
        super(GSA, self).__init__()    
        self.channels = channels
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        if prev_atns is None:
            qkv = self.qkv_dwconv(self.qkv(x))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q @ k.transpose(-2, -1)) * self.temperature
            attn = self.act(attn)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w)
            y = self.project_out(y)
            return y, attn
        else:        
            attn = prev_atns
            v = rearrange(x, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            out = (attn @ v)
            y = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
            y = rearrange(y, 'b (head c) h w -> b (c head) h w', head=self.num_heads, h=h, w=w) 
            y = self.project_out(y)
            return y


class RSA(nn.Module):
    def __init__(self, channels, num_heads, shifts=1, window_sizes=[4, 8, 12], bias=False):
        super(RSA, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.act = nn.ReLU()

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        if prev_atns is None:
            wsize = self.window_sizes
            x_ = x
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
            qkv = self.qkv_dwconv(self.qkv(x_))
            q, k, v = qkv.chunk(3, dim=1)
            q = rearrange(q, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            k = rearrange(k, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            v = rearrange(v, 'b c (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)

            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            attn = (q.transpose(-2, -1) @ k) * self.temperature # b (h w) (dh dw) (dh dw)
            attn = self.act(attn)
            out = (v @ attn)
            out = rearrange(out, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h//wsize, w=w//wsize, dh=wsize, dw=wsize)
            if self.shifts > 0:
                out = torch.roll(out, shifts=(wsize//2, wsize//2), dims=(2, 3))
            y = self.project_out(out)
            return y, attn
        else:        
            wsize = self.window_sizes
            if self.shifts > 0:
                x = torch.roll(x, shifts=(-wsize//2, -wsize//2), dims=(2,3))
            atn = prev_atns
            v = rearrange(x, 'b (c) (h dh) (w dw) -> b (h w) (dh dw) c', dh=wsize, dw=wsize)
            y_ = (v @ atn)
            y_ = rearrange(y_, 'b (h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h//wsize, w=w//wsize, dh=wsize, dw=wsize)
            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))        
            y = self.project_out(y_)
            return y


class FDT(nn.Module):
    def __init__(self, inp_channels, window_sizes, shifts, num_heads, shared_depth=1, ffn_expansion_factor=2.66):
        super(FDT, self).__init__()
        self.shared_depth = shared_depth
        
        modules_ffd = {}
        modules_att = {} 
        modules_norm = {} 
        for i in range(shared_depth):
            modules_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modules_att['att_{}'.format(i)] = RSA(channels=inp_channels, num_heads=num_heads, shifts=shifts, window_sizes=window_sizes)
            modules_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modules_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modules_ffd = nn.ModuleDict(modules_ffd)
        self.modules_att = nn.ModuleDict(modules_att)
        self.modules_norm = nn.ModuleDict(modules_norm)

        modulec_ffd = {}
        modulec_att = {} 
        modulec_norm = {} 
        for i in range(shared_depth):
            modulec_ffd['ffd{}'.format(i)] = FeedForward(inp_channels, ffn_expansion_factor, bias=False)
            modulec_att['att_{}'.format(i)] = GSA(channels=inp_channels, num_heads=num_heads)
            modulec_norm['norm_{}'.format(i)] = LayerNorm(inp_channels, 'WithBias')
            modulec_norm['norm_{}'.format(i+2)] = LayerNorm(inp_channels, 'WithBias')
        self.modulec_ffd = nn.ModuleDict(modulec_ffd)
        self.modulec_att = nn.ModuleDict(modulec_att)
        self.modulec_norm = nn.ModuleDict(modulec_norm)

    def forward(self, x):
        atn = None
        B, C, H, W = x.size()
        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), None)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x_ = self.modules_att['att_{}'.format(i)](self.modules_norm['norm_{}'.format(i)](x), atn)
                x = self.modules_ffd['ffd{}'.format(i)](self.modules_norm['norm_{}'.format(i+2)](x_ + x)) + x_

        for i in range(self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x_, atn = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), None)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
            else:
                x = self.modulec_att['att_{}'.format(i)](self.modulec_norm['norm_{}'.format(i)](x), atn)
                x = self.modulec_ffd['ffd{}'.format(i)](self.modulec_norm['norm_{}'.format(i+2)](x_ + x)) + x_
        
        return x



class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)



class WFU(nn.Module):
    def __init__(self, dim_big, dim_small):
        super(WFU, self).__init__()
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )

        self.channel_tranformation = nn.Sequential(
            nn.Conv2d(dim_big+dim_small, dim_big+dim_small // 1, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(dim_big+dim_small // 1, dim_big*3, kernel_size=1, padding=0),
        )

    def forward(self, x_big, x_small):
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))
        out = self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)

        return out


class WFD(nn.Module):
    def __init__(self, dim_in, dim, need=False):
        super(WFD, self).__init__()
        self.need = need
        if need:
            self.first_conv = nn.Conv2d(dim_in, dim, kernel_size=1, padding=0)
            self.HaarWavelet = HaarWavelet(dim, grad=False)
            self.dim = dim
        else:
            self.HaarWavelet = HaarWavelet(dim_in, grad=False)
            self.dim = dim_in

    def forward(self, x):
        if self.need:
            x = self.first_conv(x)
        
        haar = self.HaarWavelet(x, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        return a, h+v+d


class WFEN(nn.Module):
    def __init__(
        self,
        inchannel=3,
        min_ch=40,
        res_depth=6,
    ):
        super(WFEN, self).__init__()
        self.first_conv = nn.Conv2d(inchannel, min_ch, kernel_size=3, padding=1)

        self.HaarDownsample1 = WFD(min_ch, min_ch*2, True)
        self.HaarDownsample2 = WFD(min_ch*2, min_ch*2*2, True)
        self.HaarDownsample3 = WFD(min_ch*2*2, min_ch*2*2, True)

        self.TransformerDown1 = nn.Sequential(
            FDT(inp_channels=min_ch, window_sizes=8, shifts=0, num_heads=4),
            FDT(inp_channels=min_ch, window_sizes=8, shifts=0, num_heads=4),
        )                
        self.TransformerDown2 = nn.Sequential(
            FDT(inp_channels=min_ch*2, window_sizes=4, shifts=1, num_heads=4),
        )
        self.TransformerDown3 = nn.Sequential(
            FDT(inp_channels=min_ch*2*2, window_sizes=2, shifts=0, num_heads=8),
        )

        self.RB1 = nn.Sequential(
            nn.Conv2d(min_ch*2, min_ch*2, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch*2, min_ch*2, kernel_size=1, padding=0, groups=1),
        )
        self.RB2 = nn.Sequential(
            nn.Conv2d(min_ch*2*2, min_ch*2*2, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch*2*2, min_ch*2*2, kernel_size=1, padding=0, groups=1),
        )
        self.RB3 = nn.Sequential(
            nn.Conv2d(min_ch*2*2, min_ch*2*2, kernel_size=1, padding=0, groups=1),
            nn.ReLU(),
            nn.Conv2d(min_ch*2*2, min_ch*2*2, kernel_size=1, padding=0, groups=1),
        )
        
        self.Transformer0 = FDT(inp_channels=min_ch*2*2, window_sizes=1, shifts=0, num_heads=8)
        Transformer = []
        for i in range(res_depth-1):
            Transformer.append(FDT(inp_channels=min_ch*2*2, window_sizes=1, shifts=1, num_heads=8),) if i%2==0 else Transformer.append(FDT(inp_channels=min_ch*2*2, window_sizes=1, shifts=0, num_heads=8))
        self.Transformer = nn.Sequential(*Transformer)

        self.TransformerUp1 = nn.Sequential(
            FDT(inp_channels=min_ch*2*2, window_sizes=8, shifts=1, num_heads=8),
        )
        self.TransformerUp2 = nn.Sequential(
            FDT(inp_channels=min_ch*2, window_sizes=4, shifts=0, num_heads=4),
        )
        self.TransformerUp3 = nn.Sequential(
            FDT(inp_channels=min_ch, window_sizes=2, shifts=1, num_heads=4),
            FDT(inp_channels=min_ch, window_sizes=2, shifts=0, num_heads=4),
        )                

        self.HaarFeatureFusion1 = WFU(min_ch*2*2, min_ch*2*2)
        self.HaarFeatureFusion2 = WFU(min_ch*2, min_ch*2*2)
        self.HaarFeatureFusion3 = WFU(min_ch, min_ch*2)

        self.out_conv = nn.Conv2d(min_ch, inchannel, kernel_size=3, padding=1)


    def forward(self, input_img):
        x_first = self.first_conv(input_img)

        ############ encoder ############
        x1 = self.TransformerDown1(x_first) # hw:128  c:40
        x1_a, x1_hvd = self.HaarDownsample1(x1) # all hw:64 c:80

        x2 = self.TransformerDown2(x1_a) # hw:64  c:80
        x2 = x2 + self.RB1(x1_hvd) # hw:64  c:160
        x2_a, x2_hvd = self.HaarDownsample2(x2) # all hw:40 c:160

        x3 = self.TransformerDown3(x2_a) # hw:40  c:160
        x3 = x3 + self.RB2(x2_hvd)
        x3_a, x3_hvd = self.HaarDownsample3(x3) # all hw:16 c:160
        ############ encoder ############

        x_trans0 = self.Transformer0(x3_a) # hw:16 c:160
        x_trans = self.Transformer(x_trans0) # hw:16 c:160
        x_trans = x_trans + self.RB3(x3_hvd) # hw:16 c:160

        ############ decoder ############
        x_up1 = self.HaarFeatureFusion1(x3, x_trans) # hw:40 c:160
        x_1 = self.TransformerUp1(x_up1) # hw:40 c:160

        x_up2 = self.HaarFeatureFusion2(x2, x_1) # hw:64 c:80
        x_2 = self.TransformerUp2(x_up2) # hw:64 c:80

        x_up3 = self.HaarFeatureFusion3(x1, x_2) # hw:128 c:40
        x_3 = self.TransformerUp3(x_up3) # hw:32 c:40
        ############ decoder ############

        out_img = self.out_conv(x_3 + x_first)
        
        return out_img