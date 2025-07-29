# Drosophila Whole Brain Fitting

This project implements a whole-brain spiking neural network model to simulate and predict the dynamics of the Drosophila (fruit fly) brain activity.

## Data

This project requires two main datasets:

- connectome data from the FlyWire project (version 630/783): https://codex.flywire.ai/
- neural activity recordings from the Drosophila brain: https://doi.org/10.6084/m9.figshare.13349282


We have also provided preprocessed data files in the `data/` directory for convenience.

Please download the datasets (https://drive.google.com/file/d/1YeespJpoRfwS_kkH-VVuuFSJNgERcUK_/view?usp=drive_link) and place them in the appropriate directories (`data/`) as specified in the code.



## Key Features

- Loads and processes Drosophila brain connectome data
- Simulates neural activity with biologically plausible dynamics
- Predicts firing rates across brain regions (neuropils)
- Evaluates prediction accuracy using bin classification and MSE metrics
- Visualizes simulated vs. experimental neural activity

## Usage

Run the training and prediction pipeline:

```bash
python drosophila_whole_brain_fitting.py --flywire_version 630 --neural_activity_id 2017-10-30_1 --devices 0 --split 0.5 --epoc 50
```

### Command Line Arguments

- `--flywire_version`: Version of the FlyWire connectome data
- `--neural_activity_id`: ID of the neural activity recording dataset
- `--devices`: GPU device ID to use
- `--split`: Train/test split ratio
- `--epoch`: Number of epochs for training

## Evaluation

The model evaluates performance using:
- Bin accuracy: Percentage of correctly predicted firing rate bins
- MSE loss: Mean squared error between predicted and actual firing rates

## Visualization

The model generates visualizations comparing:
- Simulated neuropil firing rates
- Experimental neuropil firing rates

Figures are saved in the output directory.


## Citation 


If you use this code or data, please cite:

```text
@article {Wang2024.09.24.614728,
	author = {Wang, Chaoming and Dong, Xingsi and Ji, Zilong and Jiang, Jiedong and Liu, Xiao and Wu, Si},
	title = {BrainScale: Enabling Scalable Online Learning in Spiking Neural Networks},
	elocation-id = {2024.09.24.614728},
	year = {2025},
	doi = {10.1101/2024.09.24.614728},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728},
	eprint = {https://www.biorxiv.org/content/early/2025/07/27/2024.09.24.614728.full.pdf},
	journal = {bioRxiv}
}
```
