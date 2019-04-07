# GNN for particle track finding

This project, utilize the Graph Neural Network (GNN) for a particle track finding task.
The input data is a set of points in 3D space that represent interation betwen particles and detector. The data is taken from the [TrackML](https://competitions.codalab.org/competitions/20112) challenge.

## content:
* **[WeizmannAI](#algorithm)** - this folder contains the DNN model used for training and evaluation of the data.
* **[submit_preprocessed_data](#preprocessing)**  - folder used to preprocess all data, and retrive pairs of hits that can be used later by the DNN model.

## Algorithm

GNN inputs are list of points (hits), and list of connections (edges). Not all hits are connected with each other, there is  to increase evaluation time. The algorithm is based on 4 steps:
1. Preprocessing: Identification of good pair of hits (edges based on track selection criteria), code that based only on this section can be found in [submit_preprocessed_data](submit_preprocessed_data) folder
2. step 2
3. step 3
4. step 4

### Preprocessing






