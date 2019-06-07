# project AI

:construction: **Under development** :construction:

This project includes a list of useful codes for ML tasks. Besides, it contains several notebooks used to exercise the basics of ML.

Outline:
* **[AIcouse](#aicourse)** - jupyter notebook for simple NN
* **[GraphNeuralNetwork](#graphneuralnetwork)** - implementation of GNN for [TrackML](https://competitions.codalab.org/competitions/20112) challenge

## AIcourse
This folder includes few Jupyter notebooks with examples for linear and logistic regression, as well as image categorization using Keras

## GraphNeuralNetwork
This folder contains GNN implementation.

### Installing the package:

To setup the package, in lxplus, do the following:
```bash
mkdir source build; cd source
asetup AnalysisBase,21.2.56,here
cd $TestArea/../build
cmake ../source
make -j4
source $TestArea/../build/$AnalysisBase_PLATFORM/setup.sh
```
Each time you come back, run:
```bash
cd source; asetup
source $TestArea/../build/$AnalysisBase_PLATFORM/setup.sh
```


