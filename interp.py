### Sparse Autoencoders

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GatedSparseCrossCoderConfig:
    input_dim: int
    num_input_layers: int
    hidden_dim: int

class GatedSparseCrossCoder(nn.Module):
    """
    A sparse cross-coder with a gated activation function. Uses L1 regularization on a gate path,
    and an auxiliary loss to prevent the gate path magnitudes from going to zero.
    Based on the work of https://arxiv.org/pdf/2404.16014 and https://transformer-circuits.pub/2024/crosscoders/index.html
    """
    def __init__(self, config: GatedSparseCrossCoderConfig):
        """Initializes from GatedSparseCrossCoderConfig"""
        super(GatedSparseCrossCoder, self).__init__()
        self.config = config

        W = nn.Parameter(torch.randn(
            self.config.input_dim, 
            self.config.num_input_layers,
            self.config.hidden_dim))
        
        W_norms = self.get_l1_norms(W)
        W_normed = torch.einsum("ild, d -> ild", W, 1/W_norms) * 0.1
        
        self.W_up = nn.Parameter(W_normed.clone())
        self.W_down = nn.Parameter(W_normed.clone())
        self.b_gate = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.b_mag = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.r_mag = nn.Parameter(torch.zeros(self.config.hidden_dim))

    def get_l1_norms(self, W: torch.Tensor) -> torch.Tensor:
        """
        Gets the summed l1 norms for a weight matrix.

        Args:
            W: Weight matrix of shape (input_dim, num_input_layers, hidden_dim)
        
        Returns:
            norms: Summed l1 norms for each hidden dimension of shape (hidden_dim)
        """
        norms_per_layer = torch.norm(W, p=1, dim=0)
        norm_sum = torch.sum(norms_per_layer, dim=1)
        return norm_sum
    
    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encodes a tensor of shape (batch_size, input_dim, input_layers)

        Args:
            x: Tensor of shape (batch_size, input_dim)
        
        Returns:
        """

        preactivations = torch.einsum("bil, ild -> bd", x, self.W_up)
        gate_path = preactivations + self.b_gate
        mag_path = preactivations * F.exp(self.b_mag) + self.b_mag

        activations = torch.heaviside(gate_path, torch.tensor(0)) * F.relu(mag_path)

        return {"activations": activations, "gate_path": gate_path, "mag_path": mag_path}
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decodes a tensor of shape (batch_size, hidden_dim)

        Args:
            x: Tensor of shape (batch_size, hidden_dim)
        
        Returns:
            y: Tensor of shape (batch_size, input_dim, input_layers)
        """
        return torch.einsum("bd, dlj -> blj", x, self.W_down)
    
    def forward(self, x: torch.Tensor, target_x: torch.Tensor = None) -> dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Tensor of shape (batch_size, input_dim, input_layers)
            target_x: Tensor of shape (batch_size, input_dim, input_layers) (optional)

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "activations", "reconstruction" and optionally
            "l2_loss", "l1_loss", "aux_loss" if target_x is provided
        """

        encoding = self.encode(x)
        activations = encoding["activations"]
        gate_path = encoding["gate_path"]
        reconstruction = self.decode(activations)

        if target_x is not None:
            # All losses calculated as average over batch
            # L2 loss is just reconstruction loss
            l2_loss = F.mse_loss(reconstruction, target_x)

            # L1 loss is the average of the product of the gate path and the l1 norms of the down weights
            # Encourages sparsity in the gate path
            W_down_l1s = self.get_l1_norms(self.W_down)
            l1_loss = torch.mean(W_down_l1s * gate_path)

            # Auxiliary loss is the average of the mse between the auxiliary reconstruction and the target
            # This prevents the gate path magnitudes from going to zero due to the l1 loss
            aux_reconstruction = torch.einsum("bd, dlj -> blj", F.relu(gate_path), self.W_down.detach())
            aux_loss = F.mse_loss(aux_reconstruction, target_x)

        return {"activations": encoding["activations"], "reconstruction": reconstruction, 
                "l2_loss": l2_loss, "l1_loss": l1_loss, "aux_loss": aux_loss}