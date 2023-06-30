# DSS²: Deep Statistical Solver for Distribution System State Estimation

This repository contains code for the paper:

*B. Habib, E. Isufi, W. v. Breda, A. Jongepier and J. L. Cremer, "Deep Statistical Solver for Distribution System State Estimation," in IEEE Transactions on Power Systems, doi: 10.1109/TPWRS.2023.3290358.*

## Data
This repository includes the synthetic data used for case studies as well as the scripts developed to generate the data.

## Models and case studies

This repository contains:

- case_study.py: Main script to build a DSS2 model and try case studies on it

- fun_dss.py: Script containing the class definition of the DSS2 model and most of the helper functions

- problem_dss.py: Script defining the problem's loss function to train the model on and some problem's related parameters

- loadsampling.py: Contains helper functions to perform sampling on the load profiles to generate randon load scenarios

Some pre-trained models are available in the saved_models folder and can be load in the case_study.py file, using keras library

Data to train your own model is available in the datasets folder. It is not needed if using a pre-trained model

Data generation:
- data_gen.py: Script to set the scenarios and networks and to generate the datasets
- pp_to_dss_data.py: Contains the helper function to create a DSS2 instance from pandapower 
- npy_to_tfrecords.py: Script to get a .tfrecords format for the DSS datasets, which is the data format used in TF2 during training


Necessary packages: Tensorflow 2.x, Pandas, PandaPower, NumPy

## License
   
This work is licensed under a
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
