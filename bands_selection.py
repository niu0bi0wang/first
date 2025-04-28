import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=2, in_chans=3, embed_dim=3, 
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
        
    def forward(self, x):
        """
        x: b c h w
        return: x: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # x: b embed_dim h' w'
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    

class BandSelector(torch.nn.Module):
    def __init__(self, k_rate=0.75, heads=7, img_size=128, num_bands=7, dropout=0., 
                 selected_bands=3):
        super(BandSelector, self).__init__()
        self.img_size = img_size
        self.heads = heads
        self.num_bands = num_bands
        self.selected_bands = selected_bands  # 指定选择的波段数量

        # 通道注意力参数
        self.to_qkv = nn.Linear(num_bands, num_bands*3)
        self.scale = (num_bands // heads) ** -0.5
        
        # 波段选择参数
        self.band_score_proj = nn.Sequential(
            nn.Linear(img_size**2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
        # 初始化参数以提高稳定性
        nn.init.xavier_uniform_(self.to_qkv.weight)
        
    @torch.amp.autocast('cuda',enabled=False)  # 使用全精度计算
    def forward(self, x):
        """
        输入形状: (batch, channels, height, width) -> (b,7,128,128)
        输出形状: (batch, selected_bands, height, width)
        """
        b, c, h, w = x.shape
        
        # 空间维度展平
        x_flat = rearrange(x, 'b c h w -> b (h w) c')  # (b,16384,7)
        
        # 生成QKV并分头
        qkv = self.to_qkv(x_flat).chunk(3, dim=-1)  # 3*(b,16384,7)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 内存高效的注意力计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b, seq_len, heads, d]
        
        with torch.cuda.amp.autocast(enabled=False):  # 确保高精度计算
            attn_output = scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.,
                scale=self.scale
            )
            
        attn_output = attn_output.transpose(1, 2)  # 恢复维度 [b, heads, seq_len, d]

        # 波段权重计算 - 使用均值池化减少计算
        band_scores = self.band_score_proj(x_flat.transpose(1, 2))  # (b,7,1)
        
        # 获取分数最高的波段索引
        _, top_indices = torch.topk(band_scores.squeeze(-1), k=self.selected_bands, dim=1)  # (b,3)
        
        # 按索引从原始输入中选取波段
        # 使用高效的批处理索引选择
        batch_indices = torch.arange(b, device=x.device).view(-1, 1).expand(-1, self.selected_bands)
        
        # 使用高效的索引操作
        selected_bands = x[batch_indices.flatten(), top_indices.flatten()].view(
            b, self.selected_bands, h, w)

        return selected_bands, top_indices

# 初始化 (保留约30%空间位置)
selector = BandSelector(num_bands=7, img_size=128, k_rate=0.3)



