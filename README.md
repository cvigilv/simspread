# *De novo* prediction of drug targets and candidates by chemical similarity-guided network-based inference
This repo contains the scripts for reproducing the results showcased in Vigil, Schuller (2022) "_De novo_ prediction of drug targets and candidates by chemical similarity-guided network-based inference".

#### -- Project Status: Submitted

## Table of Contents
- Requirements
- Repository description
- Usage
- Contact

## Requirements

Python requirements:
- `python` >= 3.9
- `matplotlib` >= 3.5.1
- `seaborn`>= 0.11.2
- `scikit-learn`>= 1.0.2
- `pandas`>= 1.4.1
- `tqdm` >= 4.62.3

Julia requirements:
- `julia` >= 1.7.2
- `CUDA.jl` >= 3.8.5
- `ArgParse.jl` >= 1.1.4
- `NamedArrays.jl` >= 0.9.6

Other:
- `bash`
- `jupyter-notebook`

## Repository description
This repository has the following organization:
```
.
├── bin                 # Scripts to run predictions
│  └── predict
│     ├── 10fold
│     ├── loo
│     └── timesplit
├── data                # Datasets used in study
│  ├── chembl
│  ├── wu2017
│  └── yamanishi2008
├── results             # Results obtained
│  ├── 10fold
│  ├── 10fold_dti
│  ├── loo
│  └── timesplit
└── src                 # Scripts needed to run predictions
   ├── evaluate
   ├── modules
   └── predict
      ├── 10fold
      ├── loo
      └── timesplit
```
For each directory, a corresponding README is available for further information

## Usage

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Scripts used to generate predictions are kept [here](bin/).
3. Scripts needed for predictions are kept [here](src/).
4. Datasets used in study are kept [here](data/).

## Contact
Any question, suggestion, advice and/or help needed to reproduce results, please contact Carlos Vigil Vásquez @ cvigil2@uc.cl.
