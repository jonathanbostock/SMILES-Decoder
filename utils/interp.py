# Separate the interpretability utils from the main code

import os
import torch

class ActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, activations_path: str):
        self.activations_path = activations_path
        self.training_path = os.path.join(activations_path, "training")
        self.batch_files = [f for f in os.listdir(self.training_path) if f.endswith('.pt')]
            
    def __len__(self):
        return len(self.batch_files)
            
    def __getitem__(self, idx):
        batch_path = os.path.join(self.training_path, self.batch_files[idx])
        return torch.load(batch_path)
