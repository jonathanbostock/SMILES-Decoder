# Separate the interpretability utils from the main code

import os
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from model import JumpSAE
from typing import Callable

class ActivationsDataset(Dataset):
    def __init__(self, activations_path: str, data_type: str = "training"):
        self.activations_path = activations_path
        self.data_type = data_type
        self.data_path = os.path.join(activations_path, data_type)
        self.batch_files = [f for f in os.listdir(self.data_path) if f.endswith('.pt')]
            
    def __len__(self):
        return len(self.batch_files)
            
    def __getitem__(self, idx):
        batch_path = os.path.join(self.data_path, self.batch_files[idx])
        return torch.load(batch_path, weights_only=True)

class SAETrainer(Trainer):
    """Trainer for Sparse Autoencoders, with custom l0 scheduler"""
    def __init__(self, *, model: JumpSAE, args: TrainingArguments, train_dataset: ActivationsDataset, data_collator: Callable):
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
