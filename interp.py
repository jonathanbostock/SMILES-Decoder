### Sparse Autoencoders
from utils import SMILESTokenizer, SMILESDataset, SMILESDecoder, device
from utils import GatedSparseCrossCoder as GSC
from utils import GatedSparseCrossCoderConfig as GSCConfig

def main():
    generate_data(
        model_path="results/",
        model_layers = [],
        dataset_path="data/allmolgen_pretrain_data_train.csv",
        output_path="results/1.3M-checkpoint-50733/activations.pt"
    )

def generate_data(
        model_path: str,
        model_layers: list[str],
        dataset_path: str,
        output_path: str,
    ) -> None:
    """
    Generates a dataset of activations from a trained model.

    Args:
        model_path: Path to the trained model
        model_layers: List of layers to generate activations for
        dataset_path: Path to the dataset to generate activations for
        output_path: Path to save the generated activations
    """

    model = SMILESDecoder.from_pretrained(model_path)

if __name__ == "__main__":
    main()