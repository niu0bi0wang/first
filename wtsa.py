import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import os 
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class DyT(nn.Module):
    """动态Tanh激活函数"""
    def __init__(self, c, init_a=0.8):
        super(DyT, self).__init__()
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.c = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.ones(c))
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.a * x)
        return self.c * x + self.b


class PatchEmbed(nn.Module):
    """图像到Patch嵌入"""
    def __init__(self, img_size=128, patch_size=2, in_chans=3, embed_dim=96, 
                 norm=nn.LayerNorm):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm(embed_dim)
        
        # 初始化参数
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        

    def forward(self, x):
        """x: b c h w -> (B, num_patches, embed_dim)"""
        B, C, H, W = x.shape
        # 检查输入尺寸
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}*{W}) doesn't match model size ({self.img_size[0]}*{self.img_size[1]})"
            
        x = self.proj(x)  # x: b embed_dim h' w'
        h, w = x.shape[2:4]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class PatchUnEmbed(nn.Module):
    """Patch嵌入到图像"""
    def __init__(self, img_size=128, patch_size=4, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 用线性层替代逆卷积提高效率
        self.H, self.W = img_size // patch_size, img_size // patch_size
        
    def forward(self, x, H, W):
        """B, N, C -> B, C, H', W'"""
        B, N, C = x.shape
        # 直接重塑为空间特征
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


@torch.jit.script  # 使用JIT加速
def window_partition(x, window_size: int=4):
    """窗口分区 - 加速版本
    x: (B, H, W, C) -> windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重排并合并批次维度
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


@torch.jit.script  # 使用JIT加速
def window_reverse(windows, window_size: int, H: int, W: int):
    """窗口重组 - 加速版本
    windows: (num_windows*B, window_size, window_size, C) -> x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        # 参数初始化
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class window_sparse_attention(nn.Module):
    """窗口稀疏注意力"""
    def __init__(self, dim, k_rate, window_size=8, num_heads=4, qkv_bias=True, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk = max(1, int(window_size * window_size * k_rate))
        
        # 创建相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        
        relative_coords[:, :, 0] += window_size - 1  # 从0开始
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        self.register_buffer("relative_position_index", relative_position_index)

        # 投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化参数
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        
        # 缓存计算结果以提高效率
        self.attn_mask_cache = {}

    def forward(self, x, mask=None):
        """
        x: (B, N, C), N = window_size * window_size
        mask: (num_windows, N, N)
        """
        B, N, C = x.shape
        # 检查输入
        assert N == self.window_size * self.window_size, \
            f"Input features have wrong size, expected {self.window_size*self.window_size}, got {N}"
        
        # QKV投影 - 使用高效计算
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 缩放点积注意力
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            N, N, -1)  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 应用mask
        if mask is not None:
            # 在缓存中查找已计算的mask结果
            cache_key = f"{mask.shape[0]}_{N}"
            if cache_key in self.attn_mask_cache:
                masked_attn = self.attn_mask_cache[cache_key]
            else:
                nW = mask.shape[0]
                attn_mask = mask.unsqueeze(1).unsqueeze(0)  # 1, 1, nW, N, N
                attn_mask = attn_mask.expand(B // nW, -1, -1, -1, -1).reshape(B, self.num_heads, N, N)
                masked_attn = attn.masked_fill(attn_mask > 0, float('-inf'))
                # 缓存结果
                if len(self.attn_mask_cache) < 10:  # 限制缓存大小
                    self.attn_mask_cache[cache_key] = masked_attn
                attn = masked_attn
        
        # 选择top-k注意力权重 - 稀疏注意力核心
        topk_attn, topk_indices = torch.topk(attn, k=self.topk, dim=-1)
        
        # Softmax归一化
        topk_attn = torch.softmax(topk_attn, dim=-1)
        topk_attn = self.attn_drop(topk_attn)
        
        # 高效的收集操作
        v_shape = v.shape
        batch_size, num_heads, seq_len, head_dim = v_shape
        
        # 准备索引
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)
        batch_indices = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1, 1)
        head_indices = torch.arange(num_heads, device=x.device).view(1, num_heads, 1, 1)
        
        # 聚合
        x = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=x.device)
        for b in range(batch_size):
            for h in range(num_heads):
                for n in range(seq_len):
                    # 获取当前位置的top-k索引和注意力权重
                    indices = topk_indices[b, h, n]
                    weights = topk_attn[b, h, n]
                    
                    # 获取对应的值并加权
                    selected_values = v[b, h, indices]
                    x[b, h, n] = (weights.unsqueeze(-1) * selected_values).sum(dim=0)
        
        # 重塑并投影
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    def __init__(self, dim, input_resoluiton, num_heads, window_size, shift_size, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resoluiton = input_resoluiton  # (h,w)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # shift_size > window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = window_sparse_attention(
            dim, window_size=self.window_size, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            k_rate=0.75)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        # 计算注意力mask（如果需要）
        if self.shift_size > 0:
            H, W = self.input_resoluiton
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, window_size, window_size,1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        
        # 缓存结果以提高效率
        self.cached_shift_windows = {}

    def forward(self, x):
        H, W = self.input_resoluiton
        B, L, C = x.shape
        shortcut_x = x
        
        # 第一个归一化层
        x = self.norm1(x) 
        x = x.view(B, H, W, C)

        # 缓存key
        cache_key = f"{B}_{H}_{W}_{self.shift_size}_{self.window_size}"
        
        # 移位窗口
        if self.shift_size > 0:
            if cache_key in self.cached_shift_windows:
                shifted_x = self.cached_shift_windows[cache_key](x)
            else:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # 如果缓存没满，添加到缓存
                if len(self.cached_shift_windows) < 10:
                    def shift_func(tensor):
                        return torch.roll(tensor, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    self.cached_shift_windows[cache_key] = shift_func
        else:
            shifted_x = x

        # 窗口分区
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # 窗口反分区
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 反移位
        if self.shift_size > 0:
            if f"reverse_{cache_key}" in self.cached_shift_windows:
                x = self.cached_shift_windows[f"reverse_{cache_key}"](shifted_x)
            else:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                # 如果缓存没满，添加到缓存
                if len(self.cached_shift_windows) < 20:
                    def reverse_shift_func(tensor):
                        return torch.roll(tensor, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                    self.cached_shift_windows[f"reverse_{cache_key}"] = reverse_shift_func
        else:
            x = shifted_x
        
        # 重塑
        x = x.view(B, H * W, C)
        
        # 残差连接
        x = shortcut_x + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class basicLayer(nn.Module):
    """基本层"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resoluiton = input_resolution 
        self.depth = depth

        # 创建Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resoluiton=input_resolution, num_heads=num_heads, 
                window_size=window_size, shift_size=0, mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                drop_path=drop_path, norm_layer=norm_layer)
            ])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class swintransformer(nn.Module):
    """Swin Transformer主模型"""
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_classes=1000, 
                 embed_dim=96, depths=2, num_heads=3, window_size=4, mlp_ratio=4., 
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = depths
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        # Patch嵌入
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      norm=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 创建层
        self.layers = nn.ModuleList()
        dim = embed_dim
        resolution = (img_size//patch_size, img_size//patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]  # stochastic depth

        for i_layer in range(self.num_layers):
            layer = basicLayer(
                dim=dim,
                depth=2,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i_layer],
            )
            self.layers.append(layer)
        
        # Patch反嵌入
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, 
                                          embed_dim=embed_dim)
        
        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore  # 忽略JIT编译
    def no_weight_decay(self):
        """返回不应该应用权重衰减的参数名"""
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """返回不应该应用权重衰减的参数关键字"""
        return {'relative_position_bias_table'}
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 检查输入尺寸
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Patch嵌入
        x, h, w = self.patch_embed(x)
        x = self.pos_drop(x)
        
        # 通过各层
        for layer in self.layers:
            x = layer(x)
        
        # 归一化
        x = self.norm(x)
        
        # Patch反嵌入
        x = self.patch_unembed(x, h, w)
        
        return x


# 模型测试代码（仅在直接运行此文件时执行）
if __name__ == "__main__":
    model = swintransformer(img_size=128, patch_size=4, in_chans=3, embed_dim=96)
    x = torch.randn(4, 3, 128, 128)  # 输入形状为 (4, 3, 128, 128)
    output = model(x)
    print("输出形状:", output.shape)