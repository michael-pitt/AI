# project AI

:construction: **<font color="red"> NOTE </font>:This project is under development** :construction:

This project includes a list of useful codes for ML tasks. Besides, it contains several notebooks used to exercise the basics of ML.
## Download:
* Cloning the project
```bash
git clone https://USERNAME:PASSWORD@github.com/mpitt82/AI.git
```
* Cloning a single folder (clone a subdirectory from git repository)
```bash
git clone --depth 1 https://USERNAME:PASSWORD@github.com/mpitt82/AI.git [FOLDER]
cd [FOLDER]
git filter-branch --prune-empty --subdirectory-filter [FOLDER] HEAD
```

## Outline:
* **[AIcourse](#aicourse)** - jupyter notebook for simple ML tasks
* **[GraphNeuralNetwork](#graphneuralnetwork)** - implementation of GNN for [TrackML](https://competitions.codalab.org/competitions/20112) challenge
* **[CalorimeterSegmentation](#calorimetersegmentation)** - Image segmentation of particle images
* **[SuperResolution](#superresolution)** - Super Resolution (SR) tasks in particle detectors. Implementation of Super resolution for the task of neutral energy regression in calorimeter based particle detector.

## AIcourse
This [AIcourse](AIcourse) folder includes few Jupyter notebooks with examples for linear and logistic regression, as well as image categorization using Keras.

## GraphNeuralNetwork
The [GraphNeuralNetwork](GraphNeuralNetwork) folder contains GNN implementation used in TrackML challenge, whereas the [WeizmannAI](GraphNeuralNetwork/WeizmannAI) is the submitted code to the challenge.

## CalorimeterSegmentation
The [CalorimeterSegmentation](CalorimeterSegmentation) folder contains CNN implementation of segmentation of calorimeter images. The task is to separate hits between charged and neutral particles.

## SuperResolution
The [SuperResolution](SuperResolution) folder contains an implementation of Super-Resolution task on images obtained form particle detectors. In contrast to computer vision tasks, particle detectors are 3D cameras incorporates different sensitive layers designed to stop particles that usually undergo two types of interactions, electromagnetic and hadronic. Often, all layers have different granularity and include various types of the response function, which make the task of super-resolution to be different from the mainstream computer vision tasks.



