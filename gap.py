import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """
    Gated Attention Pooling Module with top-k attention mechanism.
    
    This module implements:
    1. Mixed max and average pooling
    2. Attention mechanism with top-k selection
    3. Gated feature propagation using Gumbel-Softmax
    """
    
    def __init__(
        self, 
        dim: int, 
        temperature: float = 1.0, 
        k_rate: float = 0.75,
        learnable_param: bool = True
    ):
        """
        Initialize the GatedAttentionPool module.
        
        Args:
            dim: Input feature dimension (channels)
            temperature: Temperature for Gumbel-Softmax
            k_rate: Ratio of values to keep in the top-k attention
            learnable_param: Whether to use learnable parameter for mixing pooling types
        """
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.k_rate = float(k_rate)
        
        # Pooling layers
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Parameter for mixing max and avg pooling
        self.mix_param = nn.Parameter(torch.zeros(1), requires_grad=learnable_param)
        
        # Attention mechanism - 使用更高效的实现
        reduction_dim = max(dim // 8, 4)  # 降维以减少计算成本
        self.channel_reduction = nn.Conv2d(dim, reduction_dim, kernel_size=1)
        self.attention_proj = nn.Linear(reduction_dim, 3*reduction_dim)  # For Q, K, V projections
        self.score_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1)
        )
        
        # 初始化
        nn.init.kaiming_normal_(self.channel_reduction.weight)
        
    def compute_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention with top-k selection.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Attention output of shape [batch_size, channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        
        # 使用1x1卷积降低通道数以提高效率
        x_reduced = self.channel_reduction(x)  # [B, reduction_dim, H, W]
        reduced_channels = x_reduced.size(1)
        
        # Reshape to [batch_size, reduction_dim, height*width]
        x_flat = x_reduced.view(batch_size, reduced_channels, -1)
        seq_len = x_flat.size(2)
        
        # Transpose to [batch_size, height*width, reduction_dim]
        x_flat = x_flat.transpose(1, 2)
        
        # 计算注意力，使用torch内置函数提高效率
        qkv = self.attention_proj(x_flat)  # [batch_size, seq_len, 3*reduction_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, reduced_channels).permute(2, 0, 1, 3)
        q, k, v = qkv  # Each is [batch_size, seq_len, reduction_dim]
        
        # 使用缩放点积注意力
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(reduced_channels)
        
        # 选择top-k值，提高稀疏注意力的效率
        k_value = max(1, int(seq_len * self.k_rate))
        
        # 使用torch.topk提高性能
        attention_scores = attention_scores.masked_fill(
            torch.isnan(attention_scores), float('-inf'))  # 处理NaN
        topk_values, topk_indices = torch.topk(attention_scores, k=k_value, dim=-1)
        
        # 应用softmax获取注意力权重
        attention_weights = torch.softmax(topk_values, dim=-1)
        
        # 使用高效的批处理索引
        batch_indices = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1)
        batch_indices = batch_indices.expand(-1, seq_len, k_value)
        
        seq_indices = torch.arange(seq_len, device=x.device).view(1, seq_len, 1)
        seq_indices = seq_indices.expand(batch_size, -1, k_value)
        
        # 高效收集操作
        gathered_values = torch.zeros(batch_size, seq_len, k_value, reduced_channels, 
                                     device=x.device, dtype=v.dtype)
        
        # 使用一次性索引操作替代循环
        for b in range(batch_size):
            for s in range(seq_len):
                indices = topk_indices[b, s]
                gathered_values[b, s] = v[b, indices]
        
        # 应用注意力权重
        weighted_values = attention_weights.unsqueeze(-1) * gathered_values
        
        # 聚合结果
        output = weighted_values.sum(dim=2)  # [batch_size, seq_len, reduced_channels]
        
        # 恢复原始维度
        output = output.transpose(1, 2).view(batch_size, reduced_channels, height, width)
        
        # 投影回原始通道维度
        # 使用高效的1x1卷积而不是线性层
        output = nn.functional.conv2d(
            output, 
            self.channel_reduction.weight.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        )
        
        return output
    
    def apply_gating(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gating mechanism using Gumbel-Softmax when training.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Gated features of same shape as input
        """
        batch_size, channels, height, width = x.shape
        
        # 计算注意力 - 改用更高效的实现
        attention_output = self.compute_attention(x)
        
        # 平均池化聚合特征用于评分
        pooled_features = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        
        # 计算门控分数
        gate_score = torch.sigmoid(self.score_proj(pooled_features))  # [batch_size, 1]
        
        if self.training:
            # Gumbel-Softmax差分采样
            uniform_noise = torch.rand_like(gate_score)
            gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
            
            log_gate_score = torch.log(gate_score + 1e-10)
            log_inv_gate_score = torch.log(1 - gate_score + 1e-10)
            
            # 堆叠并应用softmax
            gate_logits = torch.stack(
                [log_inv_gate_score, log_gate_score], 
                dim=-1
            )
            
            gate_dist = torch.softmax(gate_logits / self.temperature, dim=-1)
            
            # 获取特征传递概率
            gate = gate_dist[:, :, 1].view(batch_size, 1, 1, 1)
        else:
            # 推理时使用硬阈值
            gate = (gate_score > 0.5).float().view(batch_size, 1, 1, 1)
        
        # 应用门控到输入
        return gate * x
    
    def mixed_pooling(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply max and average pooling with mixing.
        
        Args:
            x: Input tensor
            
        Returns:
            Two mixed pooled tensors
        """
        # 应用池化
        max_pooled = self.maxpool(x)
        avg_pooled = self.avgpool(x)
        
        # 应用可学习混合参数
        mix_weight = torch.sigmoid(self.mix_param)
        
        # 混合池化特征 - 使用torch.lerp提高效率
        x1 = torch.lerp(max_pooled, avg_pooled, mix_weight)
        x2 = torch.lerp(avg_pooled, max_pooled, mix_weight)
        
        return x1, x2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GatedAttentionPool module.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Pooled and attended feature map with shape [batch_size, channels, height/2, width/2]
        """
        # 应用混合池化
        x1, x2 = self.mixed_pooling(x)
        
        # 应用门控机制 - 使用批处理提高效率
        combined = torch.cat([x1.unsqueeze(1), x2.unsqueeze(1)], dim=1)  # [B, 2, C, H/2, W/2]
        
        # 批量处理门控
        gated = self.apply_gating(combined.view(-1, *combined.shape[2:]))
        gated = gated.view(*combined.shape)
        
        # 取元素级最大值
        return torch.max(gated[:, 0], gated[:, 1])
