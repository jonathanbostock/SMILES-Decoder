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
import matplotlib.pyplot as plt

from smiles_decoder_rs import SMILESTokenizer, SMILESParser
from utils.main_utils import get_sae_collate_fn, SAETrainer, SMILESDataset
from utils.device import device
from utils.model import JumpSAE, JumpSAEConfig, JumpSAECollection, SMILESTransformer

def main():

    do_sae_training()

    #do_sae_testing()

    #do_sae_plotting()


def do_sae_training():
    sae_configs = [
        JumpSAEConfig(model_component=f"residual_post_{component_name}_{i}", input_size=256, hidden_size=4096, target_l0=32, epsilon=1e-2)
        for i in range(6)
        for component_name in ("attn", "ff")
    ] + [
        JumpSAEConfig(model_component="fingerprints", input_size=256, hidden_size=4096, target_l0=32, epsilon=1e-2)
    ]

    dataset = SMILESDataset(
        csv_path="data/allmolgen_pretrain_data_100maxlen_FIXEDCOLS_train.csv",
        tokenizer=SMILESTokenizer("allmolgen_frag_smiles_vocab.txt"),
        parser=SMILESParser("graph_vocab.txt"),
        device=device,
        max_length=512
    )
    transformer = SMILESTransformer.from_pretrained(
        model_path ="results/canonical_model/model.safetensors",
        config_path="results/canonical_model/config.json"
    ).to(device)
    transformer.eval()

    train_saes(
        sae_configs=sae_configs,
        model=transformer,
        dataset=dataset,
        output_path="interp/results/",
    )

def do_sae_testing():
    test_saes(
        sae_paths=["interp/results/4096_16", "interp/results/4096_32", "interp/results/4096_64"],
        dataset_path="interp/canonical_activations/",
        csv_path="data/allmolgen_pretrain_data_val.csv",
        output_path="interp/results/"
    )

def do_sae_plotting():
    plot_sae_results(
        results_path="interp/results/",
        sae_names=["4096_16", "4096_32", "4096_64"],
        output_path="interp/results/"
    )

def plot_sae_results(
    results_path: str,
    sae_names: list[str],
    output_path: str
) -> None:
    """
    Plots the results of the SAE testing.
    """
    for sae_name in sae_names:
        fig, ax = plt.subplots(1, 1)
        with open(os.path.join(results_path, f"{sae_name}.json"), "r") as f:
            results = json.load(f)

        total_activations = np.array(results["total_activations"])
        activation_frequencies = total_activations
        # Create logarithmically spaced bins
        min_freq = activation_frequencies[activation_frequencies > 0].min()
        max_freq = activation_frequencies.max()
        bins = np.logspace(np.log10(min_freq), np.log10(max_freq), 50)
        ax.hist(activation_frequencies, bins=bins, log=True)
        ax.set_xscale('log')
        ax.set_xlabel('Frequency of activation')
        ax.set_ylabel('Number of features')
        ax.set_title(f'Feature Activation Frequency Distribution\n{sae_name}')

        fig.tight_layout()
        fig.savefig(os.path.join(output_path, f"{sae_name}_activation_hist.png"))
        plt.close(fig)

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

    dataset = SMILESDataset(csv_path, device=device, max_length=512)
    valid_data_df = pd.read_csv(csv_path)

    for sae_index, sae_path in enumerate(sae_paths):
        print(f"Testing SAE {sae_index + 1} of {len(sae_paths)}")

        sae = JumpSAE.from_pretrained(sae_path)
        sae.to(device)
        sae.eval()

        total_items = 0

        mse_list = []
        l0_list = []

        # Get top 10 highest activating molecules for each feature
        highest_activating_molecule_indices = [
            [{"molecule_index": -i, "activation": -1} for i in range(10)]
            for _ in range(sae.config.hidden_size)]
        
        total_activations = np.zeros(sae.config.hidden_size)

        for batch_index, batch in tqdm(enumerate(dataset)):
            batch = batch.to(device)
            sae_output = sae(batch)
            activations = sae_output.feature_activations.detach().cpu().numpy()

            mse = torch.mean((sae_output.output - batch)**2).item()
            l0 = torch.mean(torch.sum((sae_output.feature_activations > 0).type(torch.float), dim=1)).item()

            mse_list.append(mse)
            l0_list.append(l0)

            ### Keep track of the highest activating molecules
            for row_index, row in enumerate(activations):
                molecule_index = batch_index * 512 + row_index
                for feature_index, feature in enumerate(row):
                    feature = float(feature)
                    if feature > highest_activating_molecule_indices[feature_index][-1]["activation"]:
                        # Add the new molecule to the list
                        # Then sort the list (highest to lowest)
                        # Then remove the last element
                        highest_activating_molecule_indices[feature_index].append({"molecule_index": molecule_index, "activation": feature})
                        highest_activating_molecule_indices[feature_index].sort(key=lambda x: x["activation"], reverse=True)
                        highest_activating_molecule_indices[feature_index] = highest_activating_molecule_indices[feature_index][:10]

            total_activations += np.sum(activations > 0, axis=0)

            total_items += activations.shape[0]

        # Convert highest activating molecules to a dictionary
        for highest_activating_molecule_list in highest_activating_molecule_indices:
            for item in highest_activating_molecule_list:
                item["molecule_index"] = int(item["molecule_index"])
                item["molecule_smiles"] = valid_data_df.iloc[item["molecule_index"]]["smiles"]

        # Save the results
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, f"{sae.config.hidden_size}_{sae.config.target_l0}.json"), "w") as f:
            json.dump({
                "average_mse": float(np.mean(mse_list)),
                "average_l0": float(np.mean(l0_list)),
                "total_activations": [float(x) / total_items for x in total_activations.tolist()],
                "highest_activating_molecules": highest_activating_molecule_indices
            }, f)

def train_saes(
    sae_configs: list[JumpSAEConfig],
    model: SMILESTransformer,
    dataset: SMILESDataset,
    output_path: str
) -> None:
    """
    Trains sparse autoencoders on a dataset of activations.
    """

    sae_collate_fn = get_sae_collate_fn(model)

    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=256,
        learning_rate=5e-5,
        weight_decay=0,
        logging_steps=100,
        save_strategy="steps",
        save_steps=10000,
        warmup_steps=500,
        remove_unused_columns=False,
        dataloader_pin_memory=False
    )

    model = JumpSAECollection(sae_configs).to(device)

    trainer = SAETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=sae_collate_fn,
        l0_warmup_fraction=0.05
    )

    trainer.train()

    model.save_pretrained(os.path.join(output_path, "final_saes"))

if __name__ == "__main__":
    main()