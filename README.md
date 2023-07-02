# BiGGS
This codebase implements Gibbs sampling with Graph-based Smoothing (GGS)

**Note:**
1. The current version of GGS requires a GPU to run.
2. To perform Graph-based Smoothing (GS), installation of the [PETSc](https://petsc.org/) and [SLEPc](https://slepc.upv.es/) libraries is required. Appropriate versions are included in the environment.yml file. Consult the previous links for further instructors
3. The trained predictors (smoothed and unsmoothed) for both AAV and GFP across all difficulties that were used to generate the results in the manuscript are available in the 'ckpt' directory

## Installation

You can install GGS by following the steps below:

```bash
clone the repository
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
- Experiments: AAV-oracle, AAV-easy-smoothed, AAV-easy-unsmoothed, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-oracle, GFP-easy-smoothed, GFP-easy-unsmoothed, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute training of a new predictor (or oracle), run the following command 
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
# If smoothed, GS needs to have been run first
python ggs/train_predictor.py experiment=train/GFP-hard-unsmoothed 
```

Running ```ggs/train_predictor.py``` uses the config file ```configs/train_predictor.yaml```. This file can be used to modify various aspects of the predictor (architecture of the CNN, batch size, validation set size, etc.). Setting ```experiment=train/GFP-hard-unsmoothed``` uses the config file ```configs/experiment/train/GFP-hard-smoothed```. This file contains the settings for the GFP hard task from the paper. Analogous files can be made to test out other task difficulties (e.g. changing the percentile cutoffs or mutation gap). The general structure here is maintained for the other experiment types. 

Any generated data files or checkpoints will be saved in the appropriate sub-directories. Currently, the name of the deepest sub-directory is the date of the run. The date is also present in the names of generated files/folders for the other experiment types when running their respective commands.

### Smoothing (**GS**)
Given an unsmoothed predictor, implements **GS**, outputting a 'smoothed.csv' file that can be used to then train a smoothed predictor. Currently, the experiment files in ```configs/experiment/GS``` use the appropriate (unsmoothed) ```last.ckpt``` file. Be sure to replace this checkpoint if you train a new unsmoothed predictor. It's also necessary that the folder containing the ```last.ckpt``` has a corresponding ```config.yaml```
- Experiments: AAV-easy, AAV-medium, AAV-hard, GFP-easy, GFP-medium, GFP-hard

To execute **GS**, run the following command
```bash
# Replace GFP-hard with any experiment from the list above 
python ggs/GS.py experiment=smooth/GFP-hard 
```

### Generating (**GWG + IE**)
Given a predictor (smoothed or unsmooted), implements **GWG + IE**, which generates new candidates. Currently, the experiment files in ```config/experiment/generate``` use the appropriate ```last.ckpt``` file. Be sure to replace this checkpoint if you train a new predictor. It's also necessary that the folder containing the ```last.ckpt``` has a corresponding ```config.yaml```
- Experiments: AAV-easy-smoothed, AAV-easy-unsmoothed, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-easy-smoothed, GFP-easy-unsmoothed, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute **GWG + IE**, run the following command
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
python ggs/GWG.py experiment=generate/GFP-hard-unsmoothed 
```
Samples will be saved to the same folder that contains ```last.ckpt```

### Evaluating (samples 128 candidates)
Given a predictor and samples, extract the best 128 candidates according to that predictor, and evaluate them based on the metrics from the manuscript: (normalized) fitness according to the oracle, diversity, and novelty. Requires training to have been run for the corresponding experiment samples as well as predictor and samples to be in the same directory. If using a new set of samples/predictor, make sure to replace the path in the appropriate experiment config file in ```configs/experiment/evaluate```
- Experiments: AAV-easy-smoothed, AAV-easy-unsmoothed, AAV-medium-smoothed, AAV-medium-unsmoothed, AAV-hard-smoothed, AAV-hard-unsmoothed, GFP-easy-smoothed, GFP-easy-unsmoothed, GFP-medium-smoothed, GFP-medium-unsmoothed, GFP-hard-smoothed, GFP-hard-unsmoothed

To execute evaluation of samples, run the following command
```bash
# Replace GFP-hard-unsmoothed with any experiment from the list above 
python ggs/evaluate.py experiment=evaluate/GFP-hard-unsmoothed 
```
Chosen candidates and evaluation metrics will be saved in same directory as the samples





