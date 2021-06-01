## Installation

It's recommended to create a virtual environment. 

To install requirements: `pip install -r requirements.txt`

To install package in editable mode: `pip install -e .`
    
## Uninstallation

You can uninstall the FDRC package with:

`python setup.py develop --uninstall`

## Experiment 1: 

You can run the experiment 1 as 

`python src/fdrc/experiment1/experiment1.py -r test_gaussian`

it will run the `test_gaussian` recipe in `EXPERIMENT1_RECIPES` and will create and 
output folder `results/experiment1/test_gaussian_TIMESTAMP` with results and Power, 
FDR and DecayFDR figures.

## Experiment 2: 

You can run the experiment 2 as 

`python src/fdrc/experiment2/experiment2.py -r test_gaussian`

it will run the `test_gaussian` recipe in `EXPERIMENT2_RECIPES` and will create and 
output folder `results/experiment2/test_gaussian_TIMESTAMP` with results and a Power-FDR
figure.