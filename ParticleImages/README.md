[:arrow_backward: BACK :arrow_backward:](mpitt82/AI)

# ParticleImages
The "Particle Images" project is used to implement different characterization tasks from calorimeter images.
Particle detector's calorimeter plays a role of image camera, but with varying granularity. 

## Outline:
* **[CalorimeterSegmentation](#calorimetersegmentation)** - Image segmentation of particle images
* **[SuperResolution](#superresolution)** - Super Resolution (SR) tasks in particle detectors. Implementation of Super resolution for the task of neutral energy regression in calorimeter based particle detector.

# Download 

To download only this folder (clone a subdirectory from git repository) exetute the following lines:
```bash
git init
git remote add origin https://USERNAME:PASSWORD@github.com/mpitt82/AI.git
git config core.sparsecheckout true
echo "ParticleImages/*" > .git/info/sparse-checkout
git pull --depth=1 origin master
```

# Data exploration notebook
Before performing different tasks on the calorimeter images, the [data exploration notebook](https://nbviewer.jupyter.org/github/mpitt82/AI/blob/master/ParticleImages/DataExploration.ipynb)
 is used to explore simulation output. In this notebook, one can find, kinematic distribution of incoming particles,
as well as the detector's energy response. A typical image obtained from all six calorimeter layers is shown in Figure 1.

![Figure 1: Calorimeter image](images/Event_200_cells.png)

[:arrow_up: TOP](#outline)
# CalorimeterSegmentation
This folder contains an implementation of segmentation of calorimeter cells.

In particle calorimetry, segmentation of cell hits concerning different particle types plays a crucial role in various tasks:
 measurement of the energy of jets, particle identification, and localization. 
A few simple approaches are tested here to separate neutral pion from charged pion hits in the detector. 
The detector hits are simulated using Geant4 simulation package with [ATLAS-simplified](https://mpitt82.github.io/Geant4-models/ATLAS-simplified) geometry (sampling calorimeter that comprises EM and Had layers).

Example of a few simulation outputs can be found in [ATLAS-simplified@cernbox](https://cernbox.cern.ch/index.php/s/oCg3en1GHAvYSTo).

[:arrow_up: TOP](#outline)
# SuperResolution
This folder contains an implementation of Super-Resolution task on images obtained form particle detectors. In contrast to computer vision tasks, particle detectors are 3D cameras incorporates different sensitive layers designed to stop particles that usually undergo two types of interactions, electromagnetic and hadronic. Often, all layers have different granularity and include various types of the response function, which make the task of super-resolution to be different from the mainstream computer vision tasks.


The detector images obtained a simplified version of Geant4 based ATLAS detector simulation ([mpitt82/Geant4-models/ATLAS-simplified](https://github.com/mpitt82/Geant4-models/tree/master/ATLAS-simplified)). 

Example of a few simulation outputs can be found in [link].

[:arrow_up: TOP](#outline)
