# CalorimeterSegmentation
This folder contains an implementation of segmentation of calorimeter cells.

In particle calorimetry, segmentation of cell hits concerning different particle types plays a crucial role in various tasks:
 measurement of the energy of jets, particle identification, and localization. 
A few simple approaches are tested here to separate neutral pion from charged pion hits in the detector. 
The detector hits are simulated using Geant4 simulation package with [ATLAS-simplified](https://mpitt82.github.io/Geant4-models/ATLAS-simplified) geometry (sampling calorimeter that comprises EM and Had layers).

Example of a few simulation outputs can be found in [ATLAS-simplified@cernbox](https://cernbox.cern.ch/index.php/s/oCg3en1GHAvYSTo).

# Download 

To download only this folder (clone a subdirectory from git repository) exetute the following lines:
```bash
git init
git remote add origin https://USERNAME:PASSWORD@github.com/mpitt82/AI.git
git config core.sparsecheckout true
echo "CalorimeterSegmentation/*" > .git/info/sparse-checkout
git pull --depth=1 origin master
```


