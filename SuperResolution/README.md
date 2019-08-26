# SuperResolution
This folder contains an implementation of Super-Resolution task on images obtained form particle detectors. In contrast to computer vision tasks, particle detectors are 3D cameras incorporates different sensitive layers designed to stop particles that usually undergo two types of interactions, electromagnetic and hadronic. Often, all layers have different granularity and include various types of the response function, which make the task of super-resolution to be different from the mainstream computer vision tasks.


The detector images obtained a simplified version of Geant4 based ATLAS detector simulation ([mpitt82/Geant4-models/ATLAS-simplified](https://github.com/mpitt82/Geant4-models/tree/master/ATLAS-simplified)). 

Example of a few simulation outputs can be found in [link].

# Download 

To download the specific folder exetute the following lines:
```bash
git clone --depth 1 https://USERNAME:PASSWORD@github.com/mpitt82/AI.git SuperResolution
cd SuperResolution
git filter-branch --prune-empty --subdirectory-filter SuperResolution HEAD
```
