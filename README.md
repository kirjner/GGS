# GGS
This codebase implements Gibbs sampling with Graph-based Smoothing v2 (GGS-v2). (v1: [arxiv](https://arxiv.org/abs/2307.00494))
The trained predictors (smoothed and unsmoothed) for both AAV and GFP across difficulties that were used to generate the results in the main text of the paper are available in the 'ckpt' directory

**NOTE1**: v1 of this codebase is available by cloning a previous commit. The current version is, however, a more streamlined implementation, and is recommended for use.

**NOTE2**: If you experience any unexpected behavior, please open an issue or contact kirjner@mit.edu

## Dependencies 

The environment.yaml file contains the necessary dependendencies to run GGS. It uses Python 3.9 along with Torch 1.12 and 
PyTorch Lightning 2.0 with [Hydra](https://github.com/facebookresearch/hydra) to run the main pipeline. The large KNN graph that is necessary for the smoothing procedure is constructed using [PyKeOps](https://www.kernel-operations.io/keops/python/installation.html).

## Installation

You can install GGS by following the steps below:

```bash
#clone the repository
git clone https://github.com/kirjner/GGS.git

# Navigate into the unzipped directory
cd GGS 

# Install and activate the conda environment from the environment.yaml file 
# then setup the ggs package
conda env create -f environment.yaml
conda activate ggs 
pip install -e .
```

## Overview

There are 4 categories of experiments that can be run, each one constituent part of the overall **GGS** pipeline:
Training, Smoothing, Generating, Evaluating

### Training
Trains a predictor, either smoothed or unsmoothed, or optionally, the oracle.

**Note:** Training a new smoothed predictor requires running **GS** (below) beforehand, to generate the necessary data files. 
- Experiments: AAV-oracle, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-oracle, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute training of a new predictor (or oracle), run the following command 
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
# If smoothed, GS needs to have been run first
python ggs/train_predictor.py experiment=train/GFP-hard-unsmoothed 
```

Running ```ggs/train_predictor.py``` uses the config file ```configs/train_predictor.yaml```. This file can be used to modify various aspects of the predictor (architecture of the CNN, batch size, validation set size, etc.). Setting ```experiment=train/GFP-hard-unsmoothed``` uses the config file ```configs/experiment/train/GFP-hard-smoothed```. This file contains the settings for the GFP hard task from the paper. Analogous files can be made to test out other task difficulties (e.g. changing the percentile cutoffs or mutation gap). The general structure here is maintained for the other experiment types. 

Any generated data files or checkpoints will be saved in the appropriate sub-directories. Currently, the name of the deepest sub-directory is the date of the run. The date is also present in the names of generated files/folders for the other experiment types when running their respective commands.

### Smoothing (**GS**)
Given an unsmoothed predictor, implements **GS**, outputting a _smoothed.csv_ file with other parameters that can be used to then train a smoothed predictor. Currently, the experiment files in ```configs/experiment/GS``` use the appropriate (unsmoothed) ```predictor.ckpt``` file. Be sure to replace this checkpoint if you train a new unsmoothed predictor. It's also necessary that the folder containing the ```predictor.ckpt``` has a corresponding ```config.yaml```
- Experiments: AAV-easy, AAV-medium, AAV-hard, GFP-easy, GFP-medium, GFP-hard. Naming may be slightly different at this time

To execute **GS**, run the following command
```bash
# Replace GFP-hard with any experiment from the list above 
python ggs/GS.py experiment=smooth/GFP-hard 
```

### Generating (**GWG**)
Given a predictor (smoothed or unsmoothed), implements **GWG** to generate new candidates. Currently, the experiment files in ```config/experiment/generate``` use the appropriate ```.ckpt``` file. Be sure to replace this checkpoint if you train a new predictor. It's also necessary that the folder containing the ```.ckpt``` has a corresponding ```config.yaml```
- Experiments: AAV-easy-smoothed, AAV-easy-unsmoothed, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-easy-smoothed, GFP-easy-unsmoothed, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute **GWG**, run the following command
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
python ggs/GWG.py experiment=generate/GFP-hard-unsmoothed 
```
Samples will be saved to the same folder that contains the ```.ckpt``` file

### Evaluating 
Given a predictor and samples, extract the best 128 candidates according to that predictor, and evaluate them based on the metrics from the manuscript: (normalized) fitness according to the oracle, diversity, and novelty. Requires training to have been run for the corresponding experiment samples as well as predictor and samples to be in the same directory. If using a new set of samples/predictor, make sure to replace the path in the appropriate experiment config file in ```configs/experiment/evaluate```
- Experiments: AAV-easy-smoothed, AAV-easy-unsmoothed, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-easy-smoothed, GFP-easy-unsmoothed, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute evaluation of samples, run the following command
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
python ggs/evaluate.py experiment=evaluate/GFP-hard-unsmoothed 
```
Chosen candidates and evaluation metrics will be saved in same directory as the samples





