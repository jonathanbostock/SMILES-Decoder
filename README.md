

# Molecular Structure Encoder/Decoder

This is a repo containing some work I've done to investigate how deep learning models encode chemical structures.
Pretraining dataset and SMILES string tokenizer vocabulary are taken from Aksamit et al. (2024).

# Data Preprocessing

Data preprocessing is done using the misc_scripts.py file. You'll need to provide a csv of SMILES strings. Aksamit et al. (2024) use a 140MB file of drug-like molecules for this.

# Training

Training is done using the `training.py` script. This script runs thte training loop for a set number of epochs.
Each SMILES string is converted into a 

# References

Aksamit, N., Tchagang, A., Li, Y. et al. Hybrid fragment-SMILES tokenization for ADMET prediction in drug discovery. BMC Bioinformatics 25, 255 (2024). https://doi.org/10.1186/s12859-024-05861-z