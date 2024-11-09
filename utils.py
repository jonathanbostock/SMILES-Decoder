### Jonathan Bostock 2024-11-09

import numpy as np
import torch
import transformers
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMILESTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        """
        Tokenizer for SMILES strings
        """
        # Initialize parent class
        super().__init__(
            pad_token="<|pad|>",
            bos_token="<|bos|>", 
            eos_token="<|eot|>",
            model_max_length=512
        )
        
        # Load the vocabulary from file
        with open("allmolgen_frag_smiles_vocab.txt", "r") as f:
            vocab = f.read().splitlines()
        
        # Add special tokens at the start
        special_tokens = ["<|pad|>", "<|bos|>", "<|split|>", "<|eot|>"]
        full_vocab = special_tokens + vocab
        
        # Create vocab dictionaries
        self.vocab = {token: i for i, token in enumerate(full_vocab)}
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, text):
        # Simple character-level tokenization for SMILES
        tokens = list(text)
        # Add special tokens in the required format
        formatted_tokens = ["<|bos|>"] + tokens + ["<|split|>"] + tokens + ["<|eot|>"]
        return formatted_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<|pad|>"])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, "<|pad|>")

class SMILESDecoder(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layer
            num_layers: Number of layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create config for GPT2 with ALiBi
        config = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=512,
            dropout=dropout,
            use_cache=True,
            position_embedding_type="alibi",
            bos_token_id=1,  # <|bos|> token
            eos_token_id=3,  # <|eot|> token
            pad_token_id=0   # <|pad|> token
        )
        
        # Initialize model components
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = transformers.GPT2Model(config)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.embedding.weight
        
    def create_causal_alibi_mask(self, batch_size, seq_length, split_positions):
        """
        Creates a causal mask that prevents tokens after <|split|> from attending to tokens before it
        
        Args:
            batch_size: Size of batch
            seq_length: Length of sequence
            split_positions: Tensor of indices where <|split|> token appears in each sequence
        """
        # Create base causal mask
        mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add split token masking
        for i in range(batch_size):
            split_pos = split_positions[i]
            mask[i, split_pos:, :split_pos] = True
            
        return mask.unsqueeze(1)  # Add head dimension
        
    def forward(self, input_ids, split_positions, attention_mask=None):
        # Create custom attention mask
        batch_size, seq_length = input_ids.shape
        causal_mask = self.create_causal_alibi_mask(
            batch_size, 
            seq_length,
            split_positions
        ).to(input_ids.device)
        
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Pass through transformer with custom mask
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            custom_causal_mask=causal_mask
        )
        
        # Get logits
        logits = self.lm_head(transformer_outputs.last_hidden_state)
        
        return logits

class _SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer, max_length=100):
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
        smiles = self.data.iloc[idx]['SMILES']
        product = self.data.iloc[idx]['Product']
        
        # Combine with special tokens
        combined = smiles + "<|split|>" + product
        
        # Tokenize
        encoding = self.tokenizer.encode(
            combined,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find split position
        split_pos = (encoding == self.tokenizer.convert_tokens_to_ids("<|split|>")).nonzero()[0][1]
        
        return {
            'input_ids': encoding.squeeze(0),
            'split_position': split_pos,
            'attention_mask': (encoding != self.tokenizer.pad_token_id).squeeze(0).float()
        }

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
    dataset = _SMILESDataset(csv_path, tokenizer)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
