### Sparse Autoencoders
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import json

from utils.training import SMILESTokenizer, SMILESDataset, SMILESTransformer, device, collate_fn
from utils.device import device
from utils.interp import ActivationsDataset, SAETrainer
from model import JumpSAE, JumpSAEConfig

def main():


    generate_data(
        model_path="results/canonical_model",
        dataset_path="data/allmolgen_pretrain_data_valid.csv",
        output_path="interp/canonical_activations/validation/",
        batch_size=512
    )


    """
    sae_configs = [
        JumpSAEConfig(input_size=256, hidden_size=4096, target_l0=target_l0, epsilon=1e-2)
        for target_l0 in [16, 32, 64]
    ]
    train_saes(
        configs=sae_configs,
        activations_path="interp/canonical_activations/",
        output_path="interp/results/",
    )
    """
    """
    test_saes(
        sae_paths=["interp/results/4096_16", "interp/results/4096_32", "interp/results/4096_64"],
        dataset_path="interp/canonical_activations/validation/",
        csv_path="data/allmolgen_pretrain_data_valid.csv",
        output_path="interp/results/"
    )
    """

def test_saes(
    sae_paths: list[str],
    dataset_path: str,
    csv_path: str,
    output_path: str
) -> None:
    """
    Test the SAEs on the validation dataset. Records the l0 of the activations and the mse.
    Also make a record of the highest activating molecules for each SAE feature, and the number of times it was activated.

    Args:
        sae_paths: Paths to the trained SAEs
        dataset_path: Path to the validation dataset
        csv_path: Path to the csv file containing the validation dataset
        output_path: Path to save the results
    """

    dataset = ActivationsDataset(dataset_path, data_type="validation")
    valid_data_df = pd.read_csv(csv_path)
    for sae_path in sae_paths:
        sae = JumpSAE.from_pretrained(sae_path)

        mse_list = []
        l0_list = []

        # Get top 10 highest activating molecules for each feature
        highest_activating_molecule_indices = [
            [{"molecule_index": -i, "activation": -1} for i in range(10)]
            for _ in range(sae.config.hidden_size)]
        
        total_activations = np.zeros(sae.config.hidden_size)

        for batch_index, batch in enumerate(dataset):
            encoding_dict = sae.encode(batch["x"])
            activations = encoding_dict["activations"].cpu().numpy()
            mse = torch.mean((activations - batch["x"])**2).item()
            l0 = torch.mean(torch.sum(encoding_dict["gate_values"], dim=1)).item()

            mse_list.append(mse)
            l0_list.append(l0)

            ### Keep track of the highest activating molecules
            for row_index, row in enumerate(activations):
                molecule_index = batch_index * 512 + row_index
                for feature_index, feature in enumerate(row):
                    if feature > highest_activating_molecule_indices[feature_index]["activation"]:
                        # Add the new molecule to the list
                        # Then sort the list (highest to lowest)
                        # Then remove the last element
                        highest_activating_molecule_indices[feature_index].append({"molecule_index": molecule_index, "activation": feature})
                        highest_activating_molecule_indices[feature_index].sort(key=lambda x: x["activation"], reverse=True)
                        highest_activating_molecule_indices[feature_index] = highest_activating_molecule_indices[feature_index][:10]

            total_activations += np.sum(activations, axis=0)

        # Convert highest activating molecules to a dictionary
        for highest_activating_molecule_list in highest_activating_molecule_indices:
            for item in highest_activating_molecule_list:
                item["molecule_index"] = int(item["molecule_index"])
                item["molecule_smiles"] = valid_data_df.iloc[item["molecule_index"]]["smiles"]

        # Save the results
        with open(os.path.join(output_path, f"{sae_path}_mse.json"), "w") as f:
            json.dump({
                "average_mse": np.mean(mse_list),
                "average_l0": np.mean(l0_list),
                "total_activations": total_activations.tolist(),
                "highest_activating_molecules": highest_activating_molecule_indices
            }, f)





def train_saes(
    configs: list[JumpSAEConfig],
    activations_path: str,
    output_path: str,
) -> None:
    """
    Trains sparse autoencoders on a dataset of activations.
    """
    # Create dataloader for activations
    dataset = ActivationsDataset(activations_path)

    for config in configs:
        sae = JumpSAE(config)
        sae.to(device)

        training_args = TrainingArguments(
            output_dir=os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"),
            num_train_epochs=5,
            per_device_train_batch_size=1,
            save_steps=1000,
            learning_rate=5e-4,
            weight_decay=0,
            logging_steps=25,
            save_strategy="steps",
            dataloader_pin_memory=False,
            max_grad_norm=1.0
        )

        trainer = SAETrainer(
            model=sae,
            args=training_args,
            train_dataset=dataset,
            data_collator = lambda x: {"x": torch.stack(x)}
        )

        trainer.train()

        # Save the model
        os.makedirs(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"), exist_ok=True)
        sae.save_pretrained(os.path.join(output_path, f"{config.hidden_size}_{config.target_l0}"))

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
    os.makedirs(output_path, exist_ok=True)
    # Don't do more than 1024 iterations, each iteration takes ~4s and generates ~30MB of data
    iterations = min(1024, len(dataset) // batch_size)
    for i in tqdm(range(iterations)):
        batch = collate_fn([dataset.__getitem__(i*batch_size + j) for j in range(batch_size)])

        with torch.no_grad():
            output = model.encode(
                batch["encoder_tokens"].to(device),
               batch["graph_distances"].to(device)
            )["fingerprints"]

        torch.save(output, f"{output_path}/batch_{i}.pt")

if __name__ == "__main__":
    main()
