### Jonathan Bostock 2024-11-19

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GraphTransformerConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float

class BiasedAttention(nn.Module):
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config
        self.W_Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_V = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_O = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = config.hidden_size ** -0.5

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)

        return self.W_O(attn @ V)

class GraphTransformerLayer(nn.Module):
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config

        self.attention = BiasedAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:

        x_normed = self.norm1(x)
        x_attn = self.attention(x_normed, x_normed, x_normed, attn_mask=attn_bias.flatten(0,1))
        x = x + self.dropout(x_attn)

        x_ff = self.feed_forward(self.norm2(x))
        x = x + self.dropout(x_ff)

        return x


class GraphTransformer(nn.Module):
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([GraphTransformerLayer(config) for _ in range(config.num_layers)])

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, attn_bias)

        return x