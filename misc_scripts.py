### Jonathan Bostock 2024-11-09

import pandas as pd
import numpy as np

def main():
    # Create train-validation-test split
    data_split(
        csv_path="data/allmolgen_pretrain_data.csv",
        seed=42,
        test_size=0.1,
        validation_size=0.1
    )
    pass

def data_split(csv_path, seed, test_size, validation_size):
    """Split a dataset into train, validation, and test sets, and save to CSV files

    Args:
        csv_path: Path to the CSV file containing the dataset
        seed: Random seed for reproducibility
        test_size: Proportion of the dataset to include in the test split
        validation_size: Proportion of the dataset to include in the validation split
    """

    # Read the CSV file
    df = pd.read_csv(csv_path, index_col=0)
    
    # Set random seed(s)
    np.random.seed(seed)
    
    # Get number of samples for each split
    n_samples = len(df)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * validation_size)
        
    # Generate random indices for splits
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test+n_val] 
    train_indices = indices[n_test+n_val:]
    
    # Split the dataframe
    test = df.iloc[test_indices]
    val = df.iloc[val_indices]
    train = df.iloc[train_indices]
    
    # Get the base filename without extension
    base_path = csv_path.rsplit('.', 1)[0]

    # Save the splits to CSV files
    train.to_csv(f"{base_path}_train.csv", index=False)
    val.to_csv(f"{base_path}_val.csv", index=False)
    test.to_csv(f"{base_path}_test.csv", index=False)

if __name__ == "__main__":
    main()