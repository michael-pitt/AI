# GNN for particle track finding

This project, utilize the Graph Neural Network (GNN) for a particle track finding task.
The input data is a set of points in 3D space that represent interation betwen particles and detector. The data is taken from the [TrackML](https://competitions.codalab.org/competitions/20112) challenge.

## content:
* **[WeizmannAI](#algorithm)** - this folder contains the DNN model used for training and evaluation of the data.
* **[submit_preprocessed_data](#preprocessing)**  - folder used to preprocess all data, and retrive pairs of hits that can be used later by the DNN model.

## Algorithm

GNN inputs are list of points (hits), and list of connections (edges). Not all hits are connected with each other. The code uses pytorch sprase matrices. The algorithm is based on 4 steps:

1. Preprocessing: Identification of good pair of hits (edges based on track selection criteria), code that based only on this section can be found in [submit_preprocessed_data](submit_preprocessed_data) folder

2. Evaluation of good pairs based on DNN. The DNN is explained in [Pre-Evaluation](#preevaluation) section.

3. Evaluation of good paris based on [GraphNN](#gnn)

4. Reconstruction of tracks based on the edges

### Preprocessing

At the pre-processing stage, good pairs of points (segments) are selected. The selection criteria was set to:
* z<sub>0</sub> cut of 100 mm
* &rho; cut of 250mm. 

&rho; is the radius of a charge particle in a magnetic field in X-Y plane originating from the origin. The radius is related to the particle transverse momentum by &rho;= p<sub>T</sub>[GeV]/(0.3&times;B[T])

For the training, the [TrackML](https://competitions.codalab.org/competitions/20112) data was preprocessed, and stored as an `npz` files. Generation of the input data for the DNN training can be found in 

### PreEvaluation

After selection of goof pair was done, based on physical cuts, further selection of edges used with DNN model. The DNN has the following structure: 

### GNN






