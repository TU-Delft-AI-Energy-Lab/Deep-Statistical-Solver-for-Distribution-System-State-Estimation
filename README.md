# DSS2
Implementation of Deep Statistical Solver for Distribution System State Estimation

This folder contains:

- case_study.py: Main script to build a DSS model and try case studies on it

- fun_dss.py: Script containing the class definition of the DSS model and most of the helper functions

- problem_dss.py: Script defining the problem's loss function to train the model on and some problem's related parameters

- loadsampling.py: Contains helper functions to perform sampling on the load profiles to generate randon load scenarios

Some pre-trained models are available in the saved_models folder and can be load in the case_study.py file, using keras library

Data to train your own model is available in the datasets folder. It is not needed if using a pre-trained model

Scripts to generate your own training data will be added in the future.


Necessary packages: Tensorflow 2.x, Pandas, PandaPower, NumPy
