# GNN for particle track finding

This project, utilize the Graph Neural Network (GNN) for a task of particle track finding.
The input data is a set of points in 3D space that represent the interaction between particles and detector. The information is taken from the [TrackML](https://competitions.codalab.org/competitions/20112) challenge.

Inspired by https://github.com/HEPTrkX/heptrkx-gnn-tracking

## content:
* **[WeizmannAI](#algorithm)** - this folder contains the DNN model used for training and evaluation of the data.
* **[submit_preprocessed_data](#preprocessing)**  - folder used to preprocess all data, and retrive pairs of hits that can be used later by the DNN model.
* **[sample_code_submission](sample_code_submission)** - folder used to evaluate GNN model. This folder calculate track reconstruction accuracy only for barrel tracks with more than 10 hits.

## Algorithm

GNN inputs are a list of points (hits), and list of connections (edges). Not all hits are connected. The code uses pytorch sprase matrices. The algorithm is based on four steps:

1. Preprocessing: Identification of good pair of hits (edges based on track selection criteria), code that based only on this section can be found in [submit_preprocessed_data](submit_preprocessed_data) folder

2. Evaluation of good pairs based on DNN. The DNN is explained in [Pre-Evaluation](#preevaluation) section.

3. Evaluation of good paris based on [GraphNN](#gnn)

4. Reconstruction of tracks based on the edges

### Preprocessing

At the pre-processing stage, good pairs of points (segments) are selected. The selection criteria were set to:
* z<sub>0</sub> cut of 100 mm
* &rho; cut of 250mm. 

&rho; is the radius of a charged particle in a magnetic field in X-Y plane originating from the origin. The radius is related to the particle transverse momentum by &rho;= p<sub>T</sub>[GeV]/(0.3&times;B[T])

For the training, the [TrackML](https://competitions.codalab.org/competitions/20112) data was preprocessed, and stored as an `npz` files. Generation of the input data for the DNN training can be found in [save_events_to_files.ipynb](https://nbviewer.jupyter.org/github/mpitt82/AI/blob/master/GraphNeuralNetwork/notebooks/save_events_to_files.ipynb)

### PreEvaluation

After the selection of good pairs (based on physical cuts), further selection of edges evaluated using a DNN model. The DNN has the following structure: 

![PreTrainModel](WeizmannAI/images/PreTrainModel.png?raw=true "PreTrainModel: for edge pre-estimation")

The output of the model is a list of edge weights which used to discriminate bad edges and reduce the input data size (the efficiency of a cut of &omega;&gt;0.2 found to be 98.5%)

### GNN

The Graph NN can be found in [WeizmannAI](WeizmannAI/) folder. 

The model contains two nets:
- Edge representation: Input parameters of each edge are processed to obtain new representation
- Message propagation: Edge weight is evaluated as following - Each edge + weighted neighbor edges are fed into NN to estimate new edge weight:

![GNN_model](WeizmannAI/images/GNN_model.png?raw=true "GNN_model: for edge classification")






