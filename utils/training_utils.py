### Jonathan Bostock 2024-11-09

import numpy as np
import torch
import pandas as pd
import json
from safetensors.torch import load_file
from dataclasses import dataclass
from tqdm import tqdm
from rdkit import Chem
import torch.nn as nn
import torch.nn.functional as F

from utils.device import device
from model import GraphTransformer, GraphTransformerConfig, Decoder, DecoderConfig, GraphTransformerOutput, DecoderOutput
from smiles_decoder_rs import SMILESTokenizer, SMILESParser

@dataclass
class SMILESTransformerConfig():
    decoder_vocab_size: int
    encoder_vocab_size: int
    hidden_size: int
    num_decoder_layers: int
    num_encoder_layers: int
    num_decoder_heads: int
    num_encoder_heads: int
    dropout: float


class SMILESTransformer(nn.Module):
    """
    Encoder-decoder transformer model which converts graphs to fingerprints and then to SMILES strings.
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

        # Create config for encoder
        encoder_config = GraphTransformerConfig(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_encoder_heads,
            dropout=self.config.dropout
        )
        
        # Create config for Decoder
        decoder_config = DecoderConfig(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_decoder_layers,
            num_heads=self.config.num_decoder_heads,
            dropout=self.config.dropout
        )
        
        # Initialize model components
        self.encoder = GraphTransformer(encoder_config)
        self.decoder = Decoder(decoder_config)

        # Create the encoder bias terms
        encoder_bias = torch.logspace(0, -3, self.config.num_encoder_heads, base=2)
        self.register_buffer("encoder_bias", encoder_bias)

        # Create shared weight tensor for embedding and output
        embedding_std = 1/np.sqrt(self.config.hidden_size)

        self.decoder_embedding = torch.nn.Parameter(torch.empty((self.config.decoder_vocab_size, self.config.hidden_size)))
        torch.nn.init.normal_(self.decoder_embedding, mean=0.0, std=embedding_std)

        self.encoder_embedding = torch.nn.Parameter(torch.empty((self.config.encoder_vocab_size, self.config.hidden_size)))
        torch.nn.init.normal_(self.encoder_embedding, mean=0.0, std=embedding_std)

    def from_pretrained(model_path: str, config_path: str) -> "SMILESTransformer":
        """
        Loads a pretrained model from a file.
        """
        state_dict = load_file(model_path)

        config = json.load(open(config_path, "r"))

        model = SMILESTransformer(
            config=SMILESTransformerConfig(
                decoder_vocab_size=config["decoder_vocab_size"],
                encoder_vocab_size=config["encoder_vocab_size"],
                hidden_size=config["hidden_size"],
                num_decoder_layers=config["num_decoder_layers"],
                num_encoder_layers=config["num_encoder_layers"],
                num_decoder_heads=config["num_decoder_heads"],
                num_encoder_heads=config["num_encoder_heads"],
                dropout=config["dropout"]
            )
        )
        model.load_state_dict(state_dict)

        return model
    
    def encode(
            self,
            input_tokens: torch.Tensor,
            graph_distances: torch.Tensor
        ) -> dict[str, GraphTransformerOutput|torch.Tensor]:
        """
        Encodes input tokens into a set of fingerprints

        Args:
            input_tokens: torch.Tensor, size (batch_size, sequence_length)
            graph_distances: torch.Tensor, size (batch_size, sequence_length, sequence_length)
        """
        embedded_tokens = self.encoder_embedding[input_tokens]

        attn_mask = (input_tokens == 0).unsqueeze(-1).expand(-1, -1, input_tokens.shape[1]).to(torch.float) * -1e9

        encoding_outputs = self.encoder(
            x=embedded_tokens,
            graph_bias=graph_distances,
            attn_mask=attn_mask
        )

        return encoding_outputs
                

    def forward(
            self,
            encoder_tokens: torch.Tensor,
            graph_distances: torch.Tensor,
            decoder_target_tokens: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
        """
        Full forward pass for training, including encoding and decoding.

        Args:
            encoder_tokens: torch.Tensor, size (batch_size, encoder_sequence_length)
            graph_distances: torch.Tensor, size (batch_size, encoder_sequence_length, encoder_sequence_length)
            decoder_target_tokens: torch.Tensor, size (batch_size, decoder_sequence_length)

        Returns:
            dict[str, torch.Tensor]: Dictionary containing "loss" and "fingerprints"
        """

        return_dict = {}

        encoding_outputs = self.encode(encoder_tokens, graph_distances)
        fingerprints = encoding_outputs.fingerprints

        embedded_tokens_for_decoder = self.decoder_embedding[decoder_target_tokens] # size (batch_size, sequence_length, hidden_size)
        embedded_tokens_for_decoder = torch.cat([fingerprints.unsqueeze(1), embedded_tokens_for_decoder], dim=1)
        decoding_outputs = self.decoder(
            x=embedded_tokens_for_decoder,
        )

        # Ignore padding tokens (0) in loss calculation
        logits = decoding_outputs.hidden_states[-1][:, :-1, :].flatten(0, 1)  # (batch*seq_len, vocab_size) 
        targets = decoder_target_tokens.flatten(0, 1)  # (batch*seq_len)
        loss = F.cross_entropy(logits, targets, ignore_index=0)

        return_dict["loss"] = loss

        return return_dict
    
    def generate(
            self,
            fingerprints: list[torch.Tensor],
            temperature: float = 1.0,
        ) -> list[dict[str, str|float|list[int]]]:
        """
        Generates SMILES strings from a list of fingerprints
        """

        return_list = []

        for fingerprint in fingerprints:
            # Ensure fingerprint is correct shape [1, 1, hidden_size]
            assert fingerprint.shape == (1, 1, self.config.hidden_size), "Fingerprints must be of shape [1, 1, hidden_size]"

            logits = fingerprint.to(device)
            total_log_prob = 0
            next_token = 1
            tokens = []

            while next_token != 2:
                logits = torch.cat([logits, self.decoder_embedding[next_token].unsqueeze(0).unsqueeze(0)], dim=1)

                decoder_output = self.decoder(logits)
                final_hidden_state = decoder_output.final_hidden_state[0, -1]
                logits = self.decoder_embedding.weight @ final_hidden_state.T
                
                if temperature != 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(logits, dim=-1).item()


                tokens.append(next_token)
                total_log_prob += torch.log(probs[0, next_token]).item()

            return_list.append({
                "smiles": self.tokenizer.decode(tokens),
                "tokens": tokens,
                "log_prob": total_log_prob
            })

        return return_list
        


class SMILESDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            csv_path: str,
            tokenizer: str|SMILESTokenizer,
            parser: str|SMILESParser,
            device: str,
            max_length:int=512
        ):
        """
        Dataset for loading SMILES strings from CSV
        
        Args:
            csv_path: Path to CSV file containing SMILES data
            tokenizer: Tokenizer to use for encoding SMILES strings
            parser: Parser to use for parsing SMILES strings
            device: Device to use for tensors
            max_length: Maximum sequence length (will pad/truncate)
        """

        if type(tokenizer) == str:
            self.tokenizer = SMILESTokenizer(vocab_path=tokenizer)
        else:
            self.tokenizer = tokenizer

        if type(parser) == str:
            self.parser = SMILESParser(vocab_path=parser)
        else:
            self.parser = parser

        self.data = pd.read_csv(csv_path)
        self.max_length = max_length

        self.return_none_if_error = True
        self.device = device
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Get an item from the dataset, which involves processing the SMILES string into
        encoder and decoder tokens, and a graph distance matrix.

        Args:
            idx: Index of the item to get
        
        Returns:
            dict: Dictionary containing "encoder_tokens", "graph_distances" and "decoder_target_tokens"
        """
        smiles = self.data.iloc[idx]['smiles']
        
        # Tokenize
        decoder_target_tokens = self.tokenizer.encode(smiles)
        atoms, distance_matrix = self.parser.parse(smiles)

        if len(decoder_target_tokens) > self.max_length:
            return None

        return_dict = {
            "encoder_tokens": torch.tensor(atoms, device=self.device, dtype=torch.long),
            "graph_distances": torch.tensor(distance_matrix, device=self.device, dtype=torch.float),
            "decoder_target_tokens": torch.tensor(decoder_target_tokens, device=self.device, dtype=torch.long)
        }

        return return_dict
        
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
    # Remove any None items from the batch
    while None in batch:
        batch.remove(None)

    output = {}

    max_encoder_len = max([len(item['encoder_tokens']) for item in batch])
    max_decoder_len = max([len(item['decoder_target_tokens']) for item in batch]) + 2

    batch_size = len(batch)

    encoder_tokens = torch.zeros(batch_size, max_encoder_len, device=batch[0]['encoder_tokens'].device).type(torch.LongTensor)
    graph_distances = torch.zeros(batch_size, max_encoder_len, max_encoder_len, device=batch[0]['graph_distances'].device)
    decoder_target_tokens = torch.zeros(batch_size, max_decoder_len, device=batch[0]['decoder_target_tokens'].device).type(torch.LongTensor)

    for i, item in enumerate(batch):
        # This is the position of the <|split|> token in new_input_ids
        encoder_tokens[i, :len(item['encoder_tokens'])] = item['encoder_tokens']

        graph_distances[i, :len(item['graph_distances']), :len(item['graph_distances'])] = item['graph_distances']

        decoder_target_tokens[i, 1:len(item['decoder_target_tokens'])+1] = item['decoder_target_tokens']
        decoder_target_tokens[i, 0] = 1
        decoder_target_tokens[i, len(item['decoder_target_tokens'])+1] = 2

    output['encoder_tokens'] = encoder_tokens
    output['graph_distances'] = graph_distances
    output['decoder_target_tokens'] = decoder_target_tokens

    return output

def _test_tokenizer():
    tokenizer = SMILESTokenizer()
    print("Encoding CCO:\n", tokenizer.encode("CCO"), "\nShould be [X, X, Y]")
    print("Decoding CCO:\n", tokenizer.decode(tokenizer.encode("CCO")), "\nShould be  C C O")
    print("Encoding C(=O)O:\n", tokenizer.encode("C(=O)O"), "\nShould be [A, B, C, D, C]")
    print("Encoding [NH-]:\n", tokenizer.encode("[NH-]"), "\nShould be [N]")

def _test_dataset():
    tokenizer = SMILESTokenizer()
    for file_location in [
        "data/allmolgen_pretrain_data_train.csv",
        "data/allmolgen_pretrain_data_val.csv",
        "data/allmolgen_pretrain_data_test.csv"
    ]:
        dataset = SMILESDataset(csv_path=file_location, tokenizer=tokenizer)
        dataset.testing_dataset_class = True
        print(f"\nTesting {file_location}:")
        for item in tqdm(dataset, desc="Testing dataset"):
            _ = item


if __name__ == "__main__":
    # Unit test the tokenizer
    # _test_tokenizer()

    # Unit test our data processor
    _test_dataset()

    pass