pyrexMD
=======

Workflow-orientated python package for Replica Exchange Molecular Dynamics.

## Installation Instructions

(1) download and install GROMACS:
https://manual.gromacs.org/documentation/current/index.html


(2) install pyrexMD:

pip3 install .


(3) enable nglview for jupyter:

jupyter nbextension enable --py widgetsnbextension

jupyter nbextension enable --py nglview


(Optional): download and install PyRosetta if you want to use pyrexMD.decoy:
http://www.pyrosetta.org/home


## Tests
To run the python tests install pytest, go into pyrexMD/tests folder and run python -m pytest

## Examples
To run one of the short examples, go into the pyrexMD/examples folder and run jupyter notebook
