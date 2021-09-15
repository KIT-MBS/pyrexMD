[![DOCS](https://img.shields.io/badge/docs-1.0-blue?&logo=github)](https://kit-mbs.github.io/pyrexMD/)
[![GH Actions CI](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml/badge.svg)](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml)
[![codecov](https://codecov.io/gh/KIT-MBS/pyrexMD/branch/master/graph/badge.svg?token=TdmhhPgQNW)](https://codecov.io/gh/KIT-MBS/pyrexMD)


About pyrexMD
=============

`pyrexMD` is mainly designed for research projects which:

- rely on (contact-guided) Replica Exchange Molecular Dynamics or (contact-guided) Molecular Dynamics
- or focus on structure analyses and comparison

It has three main goals:

1. Interactive 'all purpose' environment. By utilizing various modified
`GROMACS` and `MDAnalysis` python bindings, this package provides a `jupyter`
notebooks based environment to design, run and analyze the entire project.
2. Data vizualization is important. Most analysis functions which calculate
meaningful values such as RMSD, QValues, contact distances etc., can create
specialized figures in the same step by passing the keyword argument
``plot=True``.
3. User-friendly and simplified application. Provided functions often combine many
individual steps into larger, workflow-orientated functions with additional
automation features. It is possible to rapidly create whole setup or structure
analysis workflows within a few commands, thereby significantly enhancing
productivity and reducing the time spent at various stages of the project.

With `pyrexMD`, it becomes especially easy to create, share and reproduce
research results or transfer the work on other target structures of interest.
Furthermore, it lowers the technical boundaries for newcomers who want to do
research utilizing Replica Exchange for enhanced sampling.

`pyrexMD` should be used in `jupyter` notebooks and requires `GROMACS` to run MD simulations.


## Documentation
You can access the documentation via https://kit-mbs.github.io/pyrexMD/
<br/>and the quick guide via https://kit-mbs.github.io/pyrexMD/quick_guide.html


## Installation Instructions
(1) install MPI:

on ubuntu:

    sudo apt-get install openmpi-bin libopenmpi-dev

macOS:

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


(Optional): download and install `PyRosetta` if you want to use the pyrexMD.decoy.abinitio module:

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

## Participating
`pyrexMD` is free and open source under the MIT license. We welcome your contribution so feel free to ask questions, report bugs, fork, etc.
