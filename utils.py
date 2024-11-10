### Jonathan Bostock 2024-11-09

import numpy as np
import torch
import transformers
import pandas as pd
import json

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
        self.transformer = transformers.GPTNeoModel(config)
        # Create shared weight tensor for embedding and output
        self.embedding = torch.nn.Parameter(torch.empty((vocab_size, hidden_size)))
        torch.nn.init.normal_(self.embedding, mean=0.0, std=0.02)
        
     
    def forward(
            self,
            input_ids: torch.Tensor,
            output_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None
        ) -> dict[str, torch.Tensor]:
        """
        Neural net forward pass, if output_ids are provided, calculate loss.
        If custom attention mask is provided, use that instead of causal mask.
        Args:
            input_ids: Input IDs
            output_ids: Output IDs
            attention_mask: Attention mask
        """
        # Get embeddings
        hidden_states = self.embedding[input_ids]

        if attention_mask is None:
            attention_mask = torch.triu(torch.full((input_ids.size(1), input_ids.size(1)), -1e5), diagonal=1).to(input_ids.device)
        
        # Pass through transformer with custom mask
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=False,
            return_dict=True,
        )
        
        # Get logits
        logits = transformer_outputs.last_hidden_state @ self.embedding.T

        outputs = {"logits": logits}

        if output_ids is not None:
            # Calculate loss with extra pad token
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                output_ids.view(-1),
                ignore_index=0
            )
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
        'output_ids': [],
        'attention_mask': [],
    }

    max_len = max([len(item['input_ids']) for item in batch]) * 2 + 4
    for item in batch:
        split_position = len(item['input_ids']) + 1
        # This is the position of the <|split|> token in new_input_ids
        input_ids = item['input_ids']
        repeated_ids = torch.cat([
            torch.tensor([1]),
            input_ids,
            torch.tensor([2]),
            torch.tensor([3]),
            input_ids,
            torch.tensor([4]),
            torch.zeros(max_len - len(input_ids) * 2 - 3, device=input_ids.device)], # -3 because of extra pad token
            dim=0).type(torch.LongTensor)
        
        new_input_ids = repeated_ids[:-1]
        new_output_ids = repeated_ids[1:].clone()
        new_output_ids[:split_position+1] = 0 # Set output tokens up to split to 0, which is not included in loss

        output['input_ids'].append(new_input_ids)
        output['output_ids'].append(new_output_ids)

        # Create base causal mask with -2e5 for upper triangle
        mask = torch.triu(torch.full((max_len, max_len), -1e5), diagonal=1)
        mask[split_position+1:, :split_position] = -1e5
        output['attention_mask'].append(mask)


    output['input_ids'] = torch.stack(output['input_ids'])
    output['output_ids'] = torch.stack(output['output_ids'])
    output['attention_mask'] = torch.stack(output['attention_mask']).unsqueeze(1)

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
    # _test_tokenizer()

    # Unit test the decoder
    #_test_decoder()