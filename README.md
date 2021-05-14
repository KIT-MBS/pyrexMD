pyREX
=====

Workflow-orientated python package for Replica-Exchange Molecular Dynamics.

## Installation Instructions

(1) download and install GROAMCS:
https://manual.gromacs.org/documentation/current/index.html

(2) install pyREX:
pip3 install .

(3) enable nglview for jupyter:
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py nglview

(Optional): download and install PyRosetta if you want to use pyREX.decoy:
http://www.pyrosetta.org/home


## Tests
To run the python tests install pytest an run python -m pytest in /path/to/pyREX/tests
