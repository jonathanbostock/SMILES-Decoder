### Sparse Autoencoders
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
import os
from tqdm import tqdm
import json

from utils.training import SMILESTokenizer, SMILESDataset, SMILESTransformer, device, collate_fn
from utils.device import device
from utils.interp import ActivationsDataset
from model import JumpSAE, JumpSAEConfig

def main():

    """
    generate_data(
        model_path="results/canonical_model",
        dataset_path="data/allmolgen_pretrain_data_train.csv",
        output_path="interp/canonical_activations/",
        batch_size=512
    )
    """

    sae_configs = [
        JumpSAEConfig(input_size=256, hidden_size=hidden_size, target_l0=target_l0, epsilon=1e-3)
        for hidden_size in [1024, 2048, 4096][0:1]
        for target_l0 in [16, 32, 64, 128][0:1]
    ]

    train_saes(
        configs=sae_configs,
        activations_path="interp/canonical_activations/",
        output_path="interp/results/",
        batch_size=512
    )

def train_saes(
    configs: list[JumpSAEConfig],
    activations_path: str,
    output_path: str,
    batch_size: int
) -> None:
    """
    Trains sparse autoencoders on a dataset of activations.
    """
    # Create dataloader for activations
    dataset = ActivationsDataset(activations_path)

    for config in configs:
        sae = JumpSAE(config).to(device)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"),
            num_train_epochs=1,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            learning_rate=1e-4,
            weight_decay=0,
            logging_steps=100,
            save_strategy="steps"
        )

        trainer = Trainer(
            model=sae,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn
        )

        trainer.train()

        # Save the model
        os.makedirs(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"), exist_ok=True)
        sae.save_pretrained(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"))
        json.dump(config.model_dump(), open(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}", "config.json"), "w"))
        # Save the training logs
        trainer.save_logs(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}", "train_logs.json"))

def generate_data(
        model_path: str,
        dataset_path: str,
        output_path: str,
        batch_size: int
    ) -> None:
    """
    Generates a dataset of activations from a trained model.

    Args:
        model_path: Path to the trained model
        dataset_path: Path to the dataset to generate activations for
        output_path: Path to save the generated activations
    """

    model = SMILESTransformer.from_pretrained(
        model_path=os.path.join(model_path, "canonical_checkpoint/model.safetensors"),
        config_path=os.path.join(model_path, "config.json")
    )
    model.to(device)
    tokenizer = SMILESTokenizer()
    dataset = SMILESDataset(
        csv_path=dataset_path,
        tokenizer=tokenizer,
    )

    # Generate activations for training set
    os.makedirs(os.path.join(output_path, "training"), exist_ok=True)
    # Don't do more than 1024 iterations, each iteration takes ~4s and generates ~30MB of data
    iterations = min(1024, len(dataset) // batch_size)
    for i in tqdm(range(iterations)):
        batch = collate_fn([dataset.__getitem__(i*batch_size + j) for j in range(batch_size)])

        with torch.no_grad():
            output = model.encode(
                batch["encoder_tokens"].to(device),
               batch["graph_distances"].to(device)
            )["fingerprints"]

        torch.save(output, f"{output_path}/training/batch_{i}.pt")

if __name__ == "__main__":
    main()