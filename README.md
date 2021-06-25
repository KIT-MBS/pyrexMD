pyrexMD
=======

`pyrexMD` is designed as an interactive 'all-purpose' toolkit for research
projects which rely on (contact-guided) Replica Exchange Molecular Dynamics.
Due to its workflow-orientated design, it is possible to rapidly create whole setup or analysis workflows, thereby significantly enhancing productivity and reducing the time spent at various stages of the project. `pyrexMD` should be used in `jupyter`
notebooks and requires `GROMACS` to run MD simulations.


## Documentation
You can access the documentation via https://kit-mbs.github.io/pyrexMD/


## Installation Instructions

(1) download and install `GROMACS`:

https://manual.gromacs.org/documentation/current/index.html


(2) download and install `pyrexMD`:

    git clone https://github.com/KIT-MBS/pyrexMD.git
    cd pyrexMD
    python -m pip install .

.. Note:: setup.py is written and tested only for python 3.6 and 3.8. Alternatively
    refer to content of requirements_python3.6.txt or requirements_python3.8.txt
    based on your python version.


(3) enable trajectory viewer for `jupyter`:

    jupyter nbextension enable --py widgetsnbextension
    jupyter nbextension enable --py nglview


(Optional): download and install `PyRosetta` if you want to use pyrexMD.decoy.abinitio:

http://www.pyrosetta.org/home


## Tests
To run the python tests install pytest, go into pyrexMD/tests folder and run

    python -m pytest

## Examples
To run one of the short examples, go into the pyrexMD/examples folder and run

    jupyter notebook
