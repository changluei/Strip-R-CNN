import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

import math
import torch
import torch.nn as nn


class DirectionalGate(nn.Module):
    """
    用 H / W 两个方向的一维汇聚来生成位置敏感 gate
    输入:  x [B, C, H, W]
    输出:  g [B, C, H, W], 范围在 [0, 1]
    """
    def __init__(self, dim, reduction=16, init_gate=0.9):
        super().__init__()
        self.init_gate = init_gate
        hidden = max(8, dim // reduction)

        # 共享压缩
        self.reduce = nn.Conv2d(dim, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

        # 分别恢复 H / W 两个方向的 gate logit
        self.expand_h = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.expand_w = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

        # 让初始 gate 更偏向 1，尽量接近原始 StripBlock 的行为
        init_gate = min(max(init_gate, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(init_gate / (1.0 - init_gate))

        nn.init.constant_(self.expand_h.bias, init_logit / 2.0)
        nn.init.constant_(self.expand_w.bias, init_logit / 2.0)

    def forward(self, x):
        b, c, h, w = x.shape

        # 沿 W 聚合，保留 H 方向结构: [B, C, H, 1]
        x_h = x.mean(dim=3, keepdim=True)

        # 沿 H 聚合，保留 W 方向结构: [B, C, 1, W] -> [B, C, W, 1]
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        # 拼接成一条“方向描述序列”: [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.reduce(y))

        # 再拆回 H / W 两段
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # [B, hidden, 1, W]

        logit_h = self.expand_h(y_h)   # [B, C, H, 1]
        logit_w = self.expand_w(y_w)   # [B, C, 1, W]

        # 用两个方向的 logit 相加，再做 sigmoid
        # 比 “sigmoid后相乘” 更容易把初始 gate 设得接近 1
        gate = torch.sigmoid(logit_h + logit_w)  # broadcast -> [B, C, H, W]
        return gate


class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2, gate_reduction=16, init_gate=0.9):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv_spatial1 = nn.Conv2d(
            dim, dim,
            kernel_size=(k1, k2),
            stride=1,
            padding=(k1 // 2, k2 // 2),
            groups=dim
        )

        # 新增：放在两个 strip conv 中间的方向门控
        self.dir_gate = DirectionalGate(
            dim=dim,
            reduction=gate_reduction,
            init_gate=init_gate
        )

        self.conv_spatial2 = nn.Conv2d(
            dim, dim,
            kernel_size=(k2, k1),
            stride=1,
            padding=(k2 // 2, k1 // 2),
            groups=dim
        )

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x, return_gate=False):
        # 1) 局部上下文
        z = self.conv0(x)

        # 2) 第一方向 strip 特征
        h = self.conv_spatial1(z)

        # 3) 由第一方向特征生成位置敏感 gate
        g = self.dir_gate(h)

        # 4) 在 H 和 Z 之间做自适应融合
        #    这行等价于: h_tilde = g * h + (1 - g) * z
        #    写成残差式更直观，也更稳一点
        h_tilde = z + g * (h - z)

        # 5) 第二方向继续传播
        v = self.conv_spatial2(h_tilde)

        # 6) 点卷积混合通道
        attn = self.conv1(v)

        out = x * attn
        if return_gate:
            return out, g
        return out


class Attention(nn.Module):
    def __init__(self, d_model,k1,k2):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0.,drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim,k1,k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class StripGateNet(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], k1s=[1,1,1,1],k2s=[19,19,19,19],drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], k1=k1s[i],k2=k2s[i],drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)



    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
                        
            for m in self.modules():
                if isinstance(m, DirectionalGate):
                    init_gate = min(max(m.init_gate, 1e-4), 1.0 - 1e-4)
                    init_logit = math.log(init_gate / (1.0 - init_gate))
                    nn.init.constant_(m.expand_h.bias, init_logit / 2.0)
                    nn.init.constant_(m.expand_w.bias, init_logit / 2.0) 
        else:
            super(StripGateNet, self).init_weights()
            
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

