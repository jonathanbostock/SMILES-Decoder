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

@dataclass
class GraphTransformerOutput:
    hidden_states: tuple[torch.Tensor]
    final_hidden_state: torch.Tensor

@dataclass
class LayerOutput:
    x_post_attn: torch.Tensor
    x_post_ff: torch.Tensor

class GraphTransformerLayer(nn.Module):
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config

        self.attention = BiasedAttention(config)
        self.attn_in_norm = nn.LayerNorm(config.hidden_size)
        self.attn_out_norm = nn.LayerNorm(config.hidden_size)

        self.feed_forward = GEGLU(config)
        self.ff_in_norm = nn.LayerNorm(config.hidden_size)
        self.ff_out_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> LayerOutput:

        x_normed = self.attn_in_norm(x)
        attn_out = self.attn_out_norm(self.attention(x_normed, attn_bias))
        x_post_attn = x + self.dropout(attn_out)

        x_normed_2 = self.ff_in_norm(x_post_attn)
        ff_out = self.ff_out_norm(self.feed_forward(x_normed_2))
        x_post_ff = x_post_attn + self.dropout(ff_out)

        return LayerOutput(x_post_attn=x_post_attn, x_post_ff=x_post_ff)

class GraphTransformer(nn.Module):
    """Custom Graph Transformer because huggingface"""
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config
        self.input_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.layers = nn.ModuleList([GraphTransformerLayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

        self.register_buffer("attn_bias_factors", torch.logspace(-8, 0, steps=config.num_heads, base=2))

    def forward(self, x: torch.Tensor, graph_bias: torch.Tensor, attn_mask: torch.Tensor) -> GraphTransformerOutput:
        """Forward pass for the graph transformer, does not do embeddings
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            graph_bias: Graph bias tensor of shape (batch_size, seq_len, seq_len)
            attn_mask: Attention mask tensor of shape (batch_size, seq_len, seq_len)

        Returns:
            GraphTransformerOutput: Output of the graph transformer
        """

        attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, self.config.num_heads, -1, -1)
        attn_bias = attn_mask_expanded + torch.einsum("bqk,h->bhqk", graph_bias, self.attn_bias_factors)

        hidden_states = []
        x = self.input_norm(x)
        for layer in self.layers:
            layer_out = layer(x, attn_bias)
            x = layer_out.x_post_ff
            hidden_states.append(layer_out.x_post_attn)
            hidden_states.append(layer_out.x_post_ff)

        final_hidden_state = self.output_norm(x)

        return GraphTransformerOutput(hidden_states=tuple(hidden_states), final_hidden_state=final_hidden_state)
    
@dataclass
class DecoderConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float

@dataclass
class DecoderOutput:
    hidden_states: tuple[torch.Tensor]
    final_hidden_state: torch.Tensor

class DecoderLayer(nn.Module):
    """Decoder layer for the SMILES transformer"""
    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.config = config

        self.attn_in_norm = nn.LayerNorm(config.hidden_size)
        self.attn_out_norm = nn.LayerNorm(config.hidden_size)
        self.attention = BiasedAttention(config)

        self.ff_in_norm = nn.LayerNorm(config.hidden_size)
        self.ff_out_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = GEGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> LayerOutput:
        """Forward pass for the decoder layer"""

        x_normed = self.attn_in_norm(x)
        attn_out = self.attn_out_norm(self.attention(x_normed, attn_bias))
        x_post_attn = x + self.dropout(attn_out)

        x_normed_2 = self.ff_in_norm(x_post_attn)
        ff_out = self.ff_out_norm(self.feed_forward(x_normed_2))
        x_post_ff = x_post_attn + self.dropout(ff_out)

        return LayerOutput(x_post_attn=x_post_attn, x_post_ff=x_post_ff)

class Decoder(nn.Module):
    """Decoder for the SMILES transformer
    
    Very very simple decoder that does not do embeddings, just a stack of decoder layers.
    Returns the final hidden state (post final layer norm) and a tuple of all hidden states.
    """
    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.config = config

        self.input_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.output_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=True)

        self.register_buffer("attn_bias_factors", torch.logspace(0, -8, steps=config.num_heads, base=2))

    def forward(self, x: torch.Tensor) -> DecoderOutput:
        """Forward pass for the decoder

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            DecoderOutput: Output of the decoder
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Causal mask everything above the main diagonal with large negative value
        attn_mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) * -1e9).to(x.device)
        # Create tensor that is -1 on main diagonal, -1 below, -2 below that, etc.
        row_indices = torch.arange(seq_len).unsqueeze(0).to(x.device)
        col_indices = torch.arange(seq_len).unsqueeze(-1).to(x.device)
        alibi_base = (row_indices - col_indices).float()
        alibi_per_head = torch.einsum("qk,h->hqk", alibi_base, self.attn_bias_factors).unsqueeze(0).expand(batch_size, -1, -1, -1)

        attn_bias = attn_mask + alibi_per_head

        hidden_states = []

        x = self.input_norm(x)
        for layer in self.layers:
            layer_out = layer(x, attn_bias)
            x = layer_out.x_post_ff
            hidden_states.append(layer_out.x_post_attn)
            hidden_states.append(layer_out.x_post_ff)

        final_hidden_state = self.output_norm(x)

        return DecoderOutput(hidden_states=tuple(hidden_states), final_hidden_state=final_hidden_state)

class BiasedAttention(nn.Module):
    """
    Biased attention mechanism for the graph transformer.
    Forward takes a tensor of shape (batch_size, seq_len, hidden_size) and a bias tensor of shape (batch_size, n_heads, seq_len, seq_len).
    """
    def __init__(self, config: GraphTransformerConfig):
        super().__init__()

        self.config = config
        self.W_Q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_K = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_V = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_O = nn.Linear(config.hidden_size, config.hidden_size)

        self.scale = (config.hidden_size/config.num_heads) ** -0.5

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        Q = self.W_Q(x).reshape(x.shape[0], -1, self.config.num_heads, self.config.hidden_size // self.config.num_heads)
        K = self.W_K(x).reshape(x.shape[0], -1, self.config.num_heads, self.config.hidden_size // self.config.num_heads)
        V = self.W_V(x).reshape(x.shape[0], -1, self.config.num_heads, self.config.hidden_size // self.config.num_heads)

        # b = batch size, q = query length, k = key length, h = heads, d = head dimension
        attn = torch.einsum("bqhd,bkhd->bhqk", Q, K) * self.scale

        attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)

        attn_values = torch.einsum("bhqk,bkhd->bqhd", attn, V).flatten(-2,-1)

        return self.W_O(attn_values)

class GEGLU(nn.Module):
    """GEGLU activation function"""
    def __init__(self, config: DecoderConfig):
        super().__init__()

        self.config = config
        self.W_up = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.W_gate = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.W_down = nn.Linear(config.hidden_size * 4, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.W_up(x)
        gate = self.W_gate(x)
        return self.W_down(up* F.gelu(gate))