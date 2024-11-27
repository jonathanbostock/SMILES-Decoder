### Jonathan Bostock 2024-11-09
import torch
import os
import json
import datetime

from utils.training_utils import SMILESTransformer, SMILESTransformerConfig, SMILESTokenizer, SMILESDataset, collate_fn, device
from transformers import Trainer, TrainingArguments

def main():

    tokenizer = SMILESTokenizer()

    model_config = SMILESTransformerConfig(
        encoder_vocab_size=100,
        decoder_vocab_size=tokenizer.get_vocab_size(),
        hidden_size=256,
        num_encoder_layers=8,
        num_encoder_heads=4,
        num_decoder_layers=8,
        num_decoder_heads=4,
        dropout=0.1,
    )

    model = SMILESTransformer(model_config)

    model.to(device)

    dataset = SMILESDataset(
        csv_path="data/allmolgen_pretrain_data_train.csv",
        tokenizer=tokenizer
    )

    # Create output directory if it doesn't exist with timestamp and model parameter count
    parameter_count = sum(p.numel() for p in model.parameters())
    parameter_count_str = f"{parameter_count / 1000000:.2g}M"
    output_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{parameter_count_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model config as JSON
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config.__dict__, f, indent=4)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="steps",
        save_steps=10000,
        warmup_steps=500,
        label_smoothing_factor=0.0,
        max_grad_norm=1.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = dataset,
        data_collator=collate_fn
    )

    trainer.train()
    # Get training logs
    logs = trainer.state.log_history
    
    # Extract loss values and steps
    train_logs = [(log["step"], log["loss"]) for log in logs if "loss" in log]
    steps, losses = zip(*train_logs)
    
    # Save logs to file
    with open(os.path.join(output_dir, "train_loss.csv"), "w") as f:
        f.write("step,loss\n")
        for step, loss in zip(steps, losses):
            f.write(f"{step},{loss}\n")

if __name__ == "__main__":
    main()