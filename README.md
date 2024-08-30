# DSS²: Deep Statistical Solver for Distribution System State Estimation

This repository contains code for the paper:

*B. Habib, E. Isufi, W. v. Breda, A. Jongepier and J. L. Cremer, "Deep Statistical Solver for Distribution System State Estimation," in IEEE Transactions on Power Systems, [doi: 10.1109/TPWRS.2023.3290358](https://doi.org/10.1109/TPWRS.2023.3290358).*

## Old Code
Code has been updated for better flow and use of PyTorch. Previous code used in the paper is available in the specified folder.

## Data
This folder includes the synthetic data used for case studies.

## Models and case studies

This repository contains:

- data.py: Helper functions:
    -  to get features for nodes and edges in specified grid
    -  to calculate power flow values based on given voltages and grid features
    -  to describe the loss function for DSS2 in gsp_wls_edge
    -  to retrieve data for training and testing from pickle files

- dss2_run.py: Main script to create a GNN model, train it with WLS and testing

- networks.py: Script defining different GNN models based on PyTorch Geometric library and PowerFlowNet repository defining a GNN model for Power Flow: https://github.com/StavrosOrf/PoweFlowNet

- loadsampling.py: Contains helper functions to perform sampling on the load profiles to generate randon load scenarios

- toy_network.py: PandaPower script to create scenarios on different grids and gather a synthetic database

Necessary packages: Pytorch, Torch Geometric, Pandas, PandaPower, NumPy

## License
   
This work is licensed under a
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
