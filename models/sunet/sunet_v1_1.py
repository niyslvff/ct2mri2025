r"""
sunet_v1_1: 带条件生成，全局变形。
"""


import math
import torch
from torch import nn
from inspect import isfunction
import numpy as np
import os
import sys
try:
    from ._nn import _nn
except:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from sunet._nn import _nn

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CompleteSize(nn.Module):
    r"""全局补全输入性转
    
    Returns:
        1. 补全的张量.
    """
    def __init__(self, divide_times) -> None:
        super().__init__()
        self.divide_num = 2 ** divide_times
        
    def forward(self, x):
        input_shape = x.shape
        B, C, *other_shape = input_shape
        front_padding_num = (other_shape[0] // self.divide_num + 1) * self.divide_num - other_shape[0] if other_shape[0] % self.divide_num != 0 else 0
        left_padding_num = (other_shape[1] // self.divide_num + 1) * self.divide_num - other_shape[1] if other_shape[1] % self.divide_num != 0 else 0
        top_padding_num = (other_shape[2] // self.divide_num + 1) * self.divide_num - other_shape[2] if other_shape[2] % self.divide_num != 0 else 0 if len(other_shape) == 3 else None
        reflection_pad = nn.ReflectionPad2d((front_padding_num, 0, left_padding_num, 0)) if len(other_shape) == 2 else \
                        nn.ReflectionPad3d((top_padding_num, 0, left_padding_num, 0, front_padding_num, 0))
        return reflection_pad(x)

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, channels, dims=2):
        super().__init__()
        if dims == 3:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear")
        elif dims == 2:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            raise "wrong dims."
        self.conv = _nn.conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, channels, dims=2, kernel_size=None, stride=None, padding=None):
        super().__init__()
        self.conv = _nn.conv_nd(
            dims,
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, ch, ch_out, groups=32, dropout=0, dims=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, ch),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            _nn.conv_nd(dims, ch, ch_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, ch, ch_out, time_emb_dim=None, dropout=0, norm_groups=32, dims=2):
        super().__init__()
        self.dims = dims

        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, ch_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(ch, ch_out, groups=norm_groups, dims=dims)
        self.block2 = Block(ch_out, ch_out, groups=norm_groups, dropout=dropout, dims=dims)
        self.res_conv = _nn.conv_nd(dims,
                                    ch, ch_out, 1) if ch != ch_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            emb_out = self.mlp(time_emb)
            h += emb_out[(...,) + (None,) * (len(h.shape) - len(emb_out.shape))]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32, dims=2):
        super().__init__()

        self.n_head = n_head
        self.dims = dims

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = _nn.conv_nd(dims, in_channel, in_channel * 3, 1, bias=False)
        self.out = _nn.conv_nd(dims, in_channel, in_channel, 1)

    def forward(self, input):
        # batch, channel, height, width = input.shape
        input_shape = input.shape
        batch, channel = input_shape[0], input_shape[1]
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        # qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, *input_shape[2:])
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        if self.dims == 2:
            attn = torch.einsum(
                "bnchw, bncyx -> bnhwyx", query, key
            ).contiguous() / math.sqrt(channel)
        elif self.dims == 3:
            attn = torch.einsum(
                "bnczhw, bncmyx -> bnzhwmyx", query, key
            ).contiguous() / math.sqrt(channel)
        else:
            raise "wrong dims."
        # attn = attn.view(batch, n_head, height, width, -1)
        attn = attn.view(batch, n_head, *input_shape[2:], -1)
        attn = torch.softmax(attn, -1)
        # attn = attn.view(batch, n_head, height, width, height, width)
        attn = attn.view(batch, n_head, *input_shape[2:], *input_shape[2:])

        if self.dims == 2:
            out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        elif self.dims == 3:
            out = torch.einsum("bnzhwmyx, bncmyx -> bnczhw", attn, value).contiguous()
        else:
            raise "wrong dims."
        # out = self.out(out.view(batch, channel, height, width))
        out = self.out(out.view(batch, channel, *input_shape[2:]))

        return out + input


class CrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32, dims=2):
        super().__init__()

        self.n_head = n_head
        self.dims = dims

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = _nn.conv_nd(dims, in_channel, in_channel, 1, bias=False)
        self.kv = _nn.conv_nd(dims, in_channel, in_channel * 2, 1, bias=False)
        self.out = _nn.conv_nd(dims, in_channel, in_channel, 1)

    def forward(self, input, c):
        # batch, channel, height, width = input.shape
        input_shape = input.shape
        c_shape = c.shape
        batch, channel = input_shape[0], input_shape[1]
        assert batch == c_shape[0] and channel == c_shape[1], "doesn't matched"
        n_head = self.n_head
        head_dim = channel // n_head

        norm_q = self.norm(input)
        norm_kv = self.norm(c)
        # qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        q = self.q(norm_q).view(batch, n_head, head_dim, *input_shape[2:])
        kv = self.kv(norm_kv).view(batch, n_head, head_dim * 2, *c_shape[2:])
        # query, key, value =qkv.chunk(3, dim=2)  # bhdyx
        query = q
        key, value = kv.chunk(2, dim=2)

        if self.dims == 2:
            attn = torch.einsum(
                "bnchw, bncyx -> bnhwyx", query, key
            ).contiguous() / math.sqrt(channel)
        elif self.dims == 3:
            attn = torch.einsum(
                "bnczhw, bncmyx -> bnzhwmyx", query, key
            ).contiguous() / math.sqrt(channel)
        else:
            raise "wrong dims."
        # attn = attn.view(batch, n_head, height, width, -1)
        attn = attn.view(batch, n_head, *input_shape[2:], -1)
        attn = torch.softmax(attn, -1)
        # attn = attn.view(batch, n_head, height, width, height, width)
        attn = attn.view(batch, n_head, *input_shape[2:], *input_shape[2:])

        if self.dims == 2:
            out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        elif self.dims == 3:
            out = torch.einsum("bnzhwmyx, bncmyx -> bnczhw", attn, value).contiguous()
        else:
            raise "wrong dims."
        # out = self.out(out.view(batch, channel, height, width))
        out = self.out(out.view(batch, channel, *input_shape[2:]))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, ch, ch_out, *,
                 noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, dims=2):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            ch, ch_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, dims=dims)
        if with_attn:
            self.attn = SelfAttention(ch_out, norm_groups=norm_groups, dims=dims)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class ResnetBlocWithCrosAttn(nn.Module):
    def __init__(self, ch, ch_out, *,
                 noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, dims=2):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            ch, ch_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout, dims=dims)
        if with_attn:
            self.attn = CrossAttention(ch_out, norm_groups=norm_groups, dims=dims)

    def forward(self, x, time_emb, c):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x, c)
        return x


class UNet(nn.Module):
    """该模型可以根据条件图片生成图片。
    """
    def __init__(
            self,
            in_channel=1,
            out_channel=1,
            dims=2,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(3, 4),
            res_blocks=1,
            dropout=0,
            with_noise_level_emb=True,
            image_size=(20, 28, 20)
    ):
        super().__init__()
        self.dim = dims

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
            
        # 全局形状补全
        self.complete_x = nn.Sequential(
            CompleteSize(len(channel_mults) - 1),
            _nn.conv_nd(self.dim, in_channel // 2, in_channel // 2, kernel_size=3, padding=1),  
        )
        self.complete_c = nn.Sequential(
            CompleteSize(len(channel_mults) - 1),
            _nn.conv_nd(self.dim, in_channel // 2, in_channel // 2, kernel_size=3, padding=1),
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [_nn.conv_nd(dims, in_channel, inner_channel,
                             kernel_size=3, padding=1)]
        c_downs = [_nn.conv_nd(dims, in_channel // 2, inner_channel,
                               kernel_size=3, padding=1)]
        for ind in range(num_mults):  # 0 1 2 3 4
            is_last = (ind == num_mults - 1)
            use_attn = ind in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn, dims=dims))
                c_downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn, dims=dims))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel, dims, 3, 2, 1))
                c_downs.append(Downsample(pre_channel, dims, 3, 2, 1))
                now_res = tuple([(e - 1) // 2 + 1 for e in list(now_res)])
                feat_channels.append(pre_channel)

        self.downs = nn.ModuleList(downs)
        self.c_downs = nn.ModuleList(c_downs)

        self.cross_attention = ResnetBlocWithCrosAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                                                      norm_groups=norm_groups,
                                                      dropout=dropout, with_attn=True, dims=dims)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, dims=dims),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, dims=dims)
        ])

        ups = []
        for ind in reversed(range(num_mults)):  # 4 3 2 1 0
            is_last = (ind < 1)  # 0
            use_attn = ind in attn_res
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn, dims=dims))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel, dims))
                now_res = tuple([e * 2 for e in list(now_res)])

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups, dims=dims)
        
    @staticmethod
    def recover_shape(x, tensor_shape):
        x_shape = x.shape
        final_x = x[:, :, x_shape[2] - tensor_shape[2]:, x_shape[3] - tensor_shape[3]:, x_shape[4] - tensor_shape[4]:] if len(tensor_shape) == 5 \
            else x[:, :, x_shape[2] - tensor_shape[2]:, x_shape[3] - tensor_shape[3]:]
        return final_x

    def forward(self, x, c, time):
        original_shape = x.shape
        # 全局形状补全
        x = self.complete_x(x)
        c = self.complete_c(c)
        
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        x = torch.concat([x, c], dim=1)

        feats = []
        down_n = 0
        for layer, c_layer in zip(self.downs, self.c_downs):
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                c = c_layer(c, t)
            else:
                x = layer(x)
                c = c_layer(c)
                down_n += 1
            feats.append(x)
            # print('down', x.shape)

        x = self.cross_attention(x, t, c)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        up_n = 4
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                residual = feats.pop()
                x = layer(torch.concat((x, residual), dim=1), t)
            else:
                x = layer(x)

            # print('up', x.shape)
        
        # 恢复成原始形状
        x = self.recover_shape(x, original_shape)

        return self.final_conv(x)

    # def weights_init(self):
    #     for L in self.modules():
    #         if isinstance(L, nn.Conv3d) or isinstance(L, nn.ConvTranspose3d) or \
    #                 isinstance(L, nn.Conv2d) or isinstance(L, nn.ConvTranspose2d):
    #             torch.nn.init.normal_(L.weight, 0.0, 0.02)
    #         if isinstance(L, nn.BatchNorm3d) or isinstance(L, nn.BatchNorm2d):
    #             torch.nn.init.normal_(L.weight, 0.0, 0.02)
    #             torch.nn.init.constant_(L.bias, 0)


def is_all_zero(odd):
    return all(map(lambda x: x == 0, odd))


if __name__ == "__main__":

    test_tensor = torch.randn((1, 1, 81, 83, 110)).to("cuda:0")
    t = torch.full((1,), 1, dtype=torch.long).to("cuda:0")
    net = UNet(in_channel=2, dims=3).to("cuda:0")
    print(f"parameters num:{sum([p.numel() for p in net.parameters()]) // (1000 ** 2)}M")
    out = net(test_tensor, test_tensor, t)
    
    import matplotlib.pyplot as plt
    x = [0, 1, 2, 3, 4]
    y = [0, 1, 4, 9, 16]
    plt.plot(x, y)
    plt.show()

