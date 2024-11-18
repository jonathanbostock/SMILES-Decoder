### Jonathan Bostock 2024-11-09

import numpy as np
import torch
import transformers
import pandas as pd
import json
import os
from safetensors.torch import load_file
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMILESTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        """
        Tokenizer for SMILES strings
        """
        # Initialize parent class
        self.vocab = {}

        super().__init__(
            pad_token="<|pad|>",
            bos_token="<|bos|>", 
            eos_token="<|eot|>",
            model_max_length=512
        )
        
        # Load the vocabulary from file
        with open("allmolgen_frag_smiles_vocab.txt", "r") as f:
            vocab = f.read().splitlines()

        special_tokens = ["<|pad|>", "<|bos|>", "<|split|>", "<|new|>", "<|eot|>"]
        
        # Create vocabulary mapping
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        self.vocab.update({token: i + len(special_tokens) for i, token in enumerate(vocab)})
        
        # Create reverse mapping
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        self._build_trie()

    def _build_trie(self):
        """Build a prefix tree for faster token matching"""
        self.trie = {}
        for token in self.vocab:
            current = self.trie
            for char in token:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_end_'] = token  # Mark end of token
    
    def _tokenize(self, text):
        """
        Tokenize text using a trie-based approach
        """
        tokens = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            current = self.trie
            longest_match = None
            longest_end = i
            
            # Follow trie as far as possible from current position
            j = i
            while j < text_len and text[j] in current:
                current = current[text[j]]
                if '_end_' in current:  # Found a complete token
                    longest_match = current['_end_']
                    longest_end = j + 1
                j += 1
            
            if longest_match:  # Use the longest token found
                tokens.append(longest_match)
                i = longest_end
            else:  # No match found, use single character
                tokens.append(text[i])
                i += 1

                
        return tokens

    def get_vocab(self):
        return self.vocab.copy()
    
    def get_vocab_size(self):
        return len(self.vocab)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<|pad|>"])
    
    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "<|pad|>")

@dataclass
class SMILESTransformerConfig():
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float


class SMILESTransformer(nn.Module):
    """
    Decoder-only transformer model for SMILES strings.
    """
    def __init__(self, config: SMILESTransformerConfig):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layer
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.config = config
        
        # Create config for GPT2
        config = transformers.GPT2Config(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_layers,
            num_attention_heads=self.config.num_heads,
            attention_dropout=self.config.dropout,
            hidden_dropout=self.config.dropout,
            bos_token_id=1,  # <|bos|> token
            eos_token_id=3,  # <|eot|> token
            pad_token_id=0,  # <|pad|> token
            position_embedding_type="alibi",  # Enable ALiBi
            max_position_embeddings=512,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        
        # Initialize model components
        self.model = transformers.GPT2Model(config)

        # Create shared weight tensor for embedding and output
        self.embedding = torch.nn.Parameter(torch.empty((self.config.vocab_size, self.config.hidden_size)))
        torch.nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def from_pretrained(model_path: str, config_path: str) -> "SMILESTransformer":
        """
        Loads a pretrained model from a file.
        """
        state_dict = load_file(model_path)

        config = json.load(open(config_path, "r"))

        model = SMILESTransformer(
            config=SMILESTransformerConfig(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config["dropout"]
            )
        )
        model.load_state_dict(state_dict)

        return model
        
    def forward(
            self,
            input_ids: torch.Tensor,
            encode: bool = True,
            decode: bool = True,
            fingerprints: torch.Tensor = None
        ) -> dict[str, torch.Tensor]:
        """
        Neural net forward pass, if output_ids are provided, calculate loss.
        If custom attention mask is provided, use that instead of causal mask.
        Args:
            input_ids: Input token IDs, size (batch_size, sequence_length)
            encode: bool, whether to encode the input IDs, default True
            decode: bool, whether to decode the input IDs, default True
            fingerprints: torch.Tensor, size (batch_size, sequence_length, hidden_size), optional

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "input_ids"
            also contains "encoding_outputs" and "fingerprints" if encode is True
            also contains "decoding_outputs" if decode is True
            also contains "loss" if self.training is True

        Invalid states:
            Must encode or decode
            Must either encode or provide fingerprints
        """
        assert decode or encode, "Must encode or decode"
        assert encode or fingerprints is not None, "Must either encode or provide fingerprints"

        return_dict = {"input_ids": input_ids}

        if encode:
            input_ids_for_encoder = input_ids.clone()
            input_ids_for_encoder[input_ids == 3] = 2 # Replace <|eos|> with <|split|>
            embedded_tokens_for_encoder = self.embedding[input_ids_for_encoder] # size (batch_size, sequence_length, hidden_size)

            encoding_outputs = self.model(inputs_embeds=embedded_tokens_for_encoder)
            return_dict["encoding_outputs"] = encoding_outputs

            # Find indices of <|split|> tokens (token_id == 2)
            split_positions = (input_ids_for_encoder == 2).nonzero()[:, 1]  # Get sequence positions only
        
            # Extract embeddings from final hidden layer at split token positions
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            fingerprints = encoding_outputs.hidden_states[-1][batch_indices, split_positions]  # size (batch_size, hidden_size)
            return_dict["fingerprints"] = fingerprints

        if decode:
            input_ids_for_decoder = input_ids.clone()
            embedded_tokens_for_decoder = self.embedding[input_ids_for_decoder][:, 1:] # size (batch_size, sequence_length, hidden_size)
            decoding_outputs = self.model(inputs_embeds=embedded_tokens_for_decoder)
            return_dict["decoding_outputs"] = decoding_outputs

        if self.training:
            # Ignore padding tokens (0) in loss calculation
            logits = decoding_outputs.hidden_states[-1][:, :-1, :].flatten(0, 1)  # (batch*seq_len, vocab_size) 
            targets = input_ids_for_decoder[:, 1:].flatten(0, 1)  # (batch*seq_len)
            loss = F.cross_entropy(logits, targets, ignore_index=0)
            return_dict["loss"] = loss

        return return_dict

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        """
        Dataset for loading SMILES strings from CSV
        
        Args:
            csv_path: Path to CSV file containing SMILES data
            tokenizer: Tokenizer to use for encoding SMILES strings
            max_length: Maximum sequence length (will pad/truncate)
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['smiles']
        
        # Tokenize
        encoding = self.tokenizer.encode(
            smiles,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding.squeeze(0),
        }
    
def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for SMILES dataset

    Args:
        batch: Batch of data
        type: list[dict]
            dictionary of data with keys 'input_ids' and 'split_positions', types torch.Tensor
        
    Returns:
        dict: Dictionary of batched data
    """

    output = {
        'input_ids': [],
    }

    max_len = max([len(item['input_ids']) for item in batch]) + 2
    for item in batch:
        split_position = len(item['input_ids']) + 1
        # This is the position of the <|split|> token in new_input_ids
        input_ids = item['input_ids']
        ids = torch.cat([
            torch.tensor([1]),
            input_ids,
            torch.tensor([2]),
            torch.zeros(max_len - len(input_ids) - 2, device=input_ids.device)],
            dim=0).type(torch.LongTensor)

        output['input_ids'].append(ids)

    output['input_ids'] = torch.stack(output['input_ids'])

    return output


def get_dataloader(csv_path, tokenizer, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the SMILES dataset
    
    Args:
        csv_path: Path to CSV file
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
    """
    dataset = SMILESDataset(csv_path, tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


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
        W_normed = torch.einsum("ild, d -> ild", W, 1/W_norms)
        
        self.W_enc = nn.Parameter(W_normed.clone())
        self.W_dec = nn.Parameter(W_normed.clone())
        self.b_gate = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.b_mag = nn.Parameter(torch.zeros(self.config.hidden_dim))
        self.r_mag = nn.Parameter(torch.zeros(self.config.hidden_dim))

        self.W_dec.register_hook(self._normalize_W_dec)

    def _normalize_W_dec(self, grad: torch.Tensor|None) -> torch.Tensor|None:
        """
        Normalizes the columns of the decoder matrix as these might otherwise go to zero
        Should normally only be called as a backward hook on W_dec.
        
        Args:
            grad: Gradient of the loss with respect to the decoder matrix (optional)
        
        Returns:
            grad: Gradient of the loss with respect to the decoder matrix (optional)
        """
        W_dec_norms = self.get_l1_norms(self.W_dec)
        self.W_dec = torch.einsum("ild, d -> ild", grad, 1/W_dec_norms)
        return grad

    @staticmethod
    def get_l1_norms(W: torch.Tensor) -> torch.Tensor:
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
            dict[str, torch.Tensor]: Dictionary containing "activations", "gate_path" and "mag_path"
        """

        preactivations = torch.einsum("bil, ild -> bd", x, self.W_enc)
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
        return torch.einsum("bd, dlj -> blj", x, self.W_dec)
    
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

            # L1 loss is the average of the product of the gate path and the l1 norms of the decoder weights
            # Encourages sparsity in the gate path
            W_dec_l1s = self.get_l1_norms(self.W_dec)
            l1_loss = torch.mean(W_dec_l1s * gate_path)

            # Auxiliary loss is the average of the mse between the auxiliary reconstruction and the target
            # This prevents the gate path magnitudes from going to zero due to the l1 loss
            aux_reconstruction = torch.einsum("bd, dlj -> blj", F.relu(gate_path), self.W_dec.detach())
            aux_loss = F.mse_loss(aux_reconstruction, target_x)

        return {"activations": encoding["activations"], "reconstruction": reconstruction,
                "l2_loss": l2_loss, "l1_loss": l1_loss, "aux_loss": aux_loss}


def _test_tokenizer():
    tokenizer = SMILESTokenizer()
    print("Encoding CCO:\n", tokenizer.encode("CCO"), "\nShould be [X, X, Y]")
    print("Decoding CCO:\n", tokenizer.decode(tokenizer.encode("CCO")), "\nShould be  C C O")
    print("Encoding C(=O)O:\n", tokenizer.encode("C(=O)O"), "\nShould be [A, B, C, D, C]")
    print("Encoding [NH-]:\n", tokenizer.encode("[NH-]"), "\nShould be [N]")

def _test_decoder():
    decoder = SMILESDecoder(vocab_size=100, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1)
    mask = decoder.create_causal_mask(1, 10, torch.tensor([5]))
    print("Creating mask:\n", mask)

if __name__ == "__main__":
    # Unit test the tokenizer
    # _test_tokenizer()

    # Unit test the decoder
    #_test_decoder()
    pass