### Jonathan Bostock 2024-11-09

import numpy as np
import torch
import pandas as pd
import json
from safetensors.torch import load_file
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from typing import Callable

from utils.device import device
from model import GraphTransformer, GraphTransformerConfig, Decoder, DecoderConfig, GraphTransformerOutput, DecoderOutput
from smiles_decoder_rs import SMILESTokenizer, SMILESParser

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

def get_collate_fn(model: SMILESTransformer) -> Callable:

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        encoder_input_ids = torch.stack([item["encoder_tokens"] for item in batch])
        graph_bias = torch.stack([item["graph_distances"] for item in batch])

        encoder_outputs = model.encode(encoder_input_ids, graph_bias)

        return encoder_outputs

    return collate_fn

class SAETrainer(Trainer):
    """Trainer for Sparse Autoencoders, with custom l0 scheduler"""
    def __init__(self, *, model: JumpSAECollection, args: TrainingArguments, train_dataset: ActivationsDataset, data_collator: Callable):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        max_steps = len(train_dataset) * args.num_train_epochs

        self.l0_scheduler = lambda step: min(step / max_steps * 5, 1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=1):
        outputs = model(**inputs)
        mse_loss = outputs.mse_loss
        l0_loss = outputs.l0_loss

        l0_coefficient = self.l0_scheduler(self.state.global_step)
        loss = mse_loss + l0_coefficient * l0_loss

        return (loss, outputs) if return_outputs else loss