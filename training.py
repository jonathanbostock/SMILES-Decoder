### Jonathan Bostock 2024-11-09
import torch

from utils import SMILESDecoder, SMILESTokenizer, SMILESDataset, collate_fn, device
from transformers import Trainer, TrainingArguments

def main():

    model = SMILESDecoder(
        vocab_size=512,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    )

    model.to(device)

    dataset = SMILESDataset(
        csv_path="data/allmolgen_pretrain_data_train.csv",
        tokenizer=SMILESTokenizer()
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        learning_rate=2e-3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        warmup_steps=500,
        label_smoothing_factor=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = dataset,
        data_collator=collate_fn
    )

    trainer.train()

if __name__ == "__main__":
    main()