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

        special_tokens = ["<|pad|>", "<|bos|>", "<|eot|>", "<|split|>"]
        
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


        # Create config for GPTNeo
        config = transformers.GPTNeoConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attention_types=[[["global"], num_layers]],
            window_size=None,
            num_attention_heads=num_heads,
            max_position_embeddings=512,
            attention_dropout=dropout,
            hidden_dropout=dropout,
            bos_token_id=1,  # <|bos|> token
            eos_token_id=3,  # <|eot|> token
            pad_token_id=0   # <|pad|> token
        )
        
        # Initialize model components
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = transformers.GPTNeoModel(config)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between embedding and output layer
        self.lm_head.weight = self.embedding.weight
        
    def create_causal_mask(self, batch_size, seq_length, split_positions):
        """
        Creates a causal mask that prevents tokens after <|split|> from attending to tokens before it.
        Returns a mask tensor with -1e5 for positions that should not attend to each other.
        
        Args:
            batch_size: Size of batch
            seq_length: Length of sequence
            split_positions: Tensor of indices where <|split|> token appears in each sequence
            
        Returns:
            torch.Tensor: Attention mask of shape (batch_size, seq_length, seq_length) with -1e5 
                         for positions that should not attend to each other
        """
        # Create base causal mask with -1e5 for upper triangle
        mask = torch.zeros((batch_size, seq_length, seq_length))
        causal_mask = torch.triu(torch.ones((seq_length, seq_length)), diagonal=1)
        mask = mask.masked_fill(causal_mask.bool(), -1e5)
        
        # Add split token masking
        for i in range(batch_size):
            split_pos = split_positions[i]
            # Fill with -1e5 where tokens after split should not attend to tokens before split
            mask[i, split_pos+1:, :split_pos] = -1e5
            
        return mask.unsqueeze(1)
        
    def forward(self, input_ids, split_positions):
        # Create custom attention mask
        batch_size, seq_length = input_ids.shape
        causal_mask = self.create_causal_mask(
            batch_size, 
            seq_length,
            split_positions
        ).to(input_ids.device)
        
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Pass through transformer with custom mask
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=causal_mask,
            head_mask=None,
            output_attentions=False,
            return_dict=True,
        )
        
        # Get logits
        logits = self.lm_head(transformer_outputs.last_hidden_state)

        outputs = {"logits": logits}

        if self.training:
            # Calculate loss with extra pad token
            target = (torch.cat([input_ids[:,1:], torch.zeros(input_ids.size(0), 1, device=input_ids.device)], dim=1)
                      .type(torch.LongTensor)
                      .to(input_ids.device))
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            outputs["loss"] = loss

        return outputs


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
    
def collate_fn(batch: list[dict]):
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
        'split_positions': []
    }

    max_len = max([len(item['input_ids']) for item in batch]) * 2 + 3
    for item in batch:
        split_position = len(item['input_ids']) + 1
        input_ids = item['input_ids']
        new_input_ids = torch.cat([
            torch.tensor([1]),
            input_ids,
            torch.tensor([2]),
            input_ids,
            torch.tensor([3]),
            torch.zeros(max_len - len(input_ids) * 2 - 3, device=input_ids.device)],
            dim=0).type(torch.LongTensor)
        output['input_ids'].append(new_input_ids)
        output['split_positions'].append(split_position)

    output['input_ids'] = torch.stack(output['input_ids'])
    output['split_positions'] = torch.tensor(output['split_positions'])

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
    _test_tokenizer()

    # Unit test the decoder
    #_test_decoder()