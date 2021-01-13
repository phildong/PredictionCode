# PredictionCode

This repository contains code and visualizations needed to reproduce the results presented in ["Decoding locomotion from population neural activity in moving *C. elegans*"](https://doi.org/10.1101/445643).

The associated data can be found [here](https://osf.io/dpr3h).

## Getting started
To reproduce the figures from the paper, first download the data from the OSF repository. Each dataset is indentified by a strain name and condition, e.g. `AML32_moving`, and is wrapped in a `.tar.gz` archive. After extracting an archive, you will find subfolders corresponding to individual recordings. Additionally, a text file, e.g. `AML32_moving_datasets.txt`, indicates which datasets should be used for analysis. For recordings of the AML310 strain, these files also contain the volume number at which the BFP laser was turned on to identify AVA. The part of the recording after this volume number is excluded for behavior analysis.

Within each dataset folder, there are five files:
* centerline.mat
* heatData.mat
* heatDataMS.mat
* pointStatsNew.mat
* positionDataMS.mat
These are the output of the neuron registration pipeline described [here](https://doi.org/10.1371/journal.pcbi.1005517).

Next, edit `utility/userTracker.py` to add your hostname to the `dataPaths` and `codePaths` dictionaries. To find your hostname, run `hostname` in a terminal.

