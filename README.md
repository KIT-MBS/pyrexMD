[![DOCS](https://img.shields.io/badge/docs-1.0-blue?&logo=github)](https://kit-mbs.github.io/pyrexMD/)
[![GH Actions CI](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml/badge.svg)](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml)
[![codecov](https://codecov.io/gh/KIT-MBS/pyrexMD/branch/master/graph/badge.svg?token=TdmhhPgQNW)](https://codecov.io/gh/KIT-MBS/pyrexMD)


About pyrexMD
=============

`pyrexMD` is designed as an interactive 'all-purpose' toolkit for research
projects which rely on (contact-guided) Replica Exchange Molecular Dynamics.
Due to its workflow-orientated design, it is possible to rapidly create whole setup or analysis workflows, thereby significantly enhancing productivity and reducing the time spent at various stages of the project. `pyrexMD` should be used in `jupyter`
notebooks and requires `GROMACS` to run MD simulations.


## Documentation
You can access the documentation via https://kit-mbs.github.io/pyrexMD/
<br/>and the quick guide via https://kit-mbs.github.io/pyrexMD/quick_guide.html


## Installation Instructions
(1) install MPI:

    # on ubuntu:
    sudo apt-get install openmpi-bin libopenmpi-dev

    # on macOS:
    brew install mpich

(2) download and install MPI-enabled version of `GROMACS` following instructions:

https://manual.gromacs.org/documentation/current/index.html


(3) download and install `pyrexMD`:

    git clone https://github.com/KIT-MBS/pyrexMD.git
    cd pyrexMD
    python -m pip install .


(4) enable trajectory viewer for `jupyter`:

    jupyter nbextension enable --py widgetsnbextension
    jupyter nbextension enable --py nglview


(Optional): download and install `PyRosetta` if you want to use pyrexMD.decoy.abinitio:

http://www.pyrosetta.org/home


## Tests
To run the python tests with code coverage, go into pyrexMD folder and run

    coverage run -m pytest
    coverage report html

and open htmlcov with any browser to see code coverage, e.g.

    firefox htmlcov/index.html


## Examples
To run one of the short examples, go into the pyrexMD/examples folder and run

    jupyter notebook

## Contributing to pyrexMD
`pyrexMD` is free and open source under the MIT license. Feel free to ask questions,
report bugs, fork, etc.
