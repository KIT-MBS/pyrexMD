[![DOCS](https://img.shields.io/badge/docs-1.0-blue?&logo=github)](https://kit-mbs.github.io/pyrexMD/)
[![GH Actions CI](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml/badge.svg)](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml)
[![codecov](https://codecov.io/gh/KIT-MBS/pyrexMD/branch/master/graph/badge.svg?token=TdmhhPgQNW)](https://codecov.io/gh/KIT-MBS/pyrexMD)


About pyrexMD
=============

`pyrexMD` is mainly designed for research projects which

- use (contact-guided) Replica Exchange Molecular Dynamics or (contact-guided) Molecular Dynamics (MD)
- or focus on structure analyses and comparison.

It has three main goals:

1. Interactive 'all-purpose' environment. By including various modified
`GROMACS` and `MDAnalysis` python bindings, this package provides a
comprehensive `jupyter`-notebooks based environment to design, run, and analyze
MD simulation projects from beginning to end.
2. Data visualization is important. In `pyrexMD`, most analysis functions for
calculating useful quantities, such as RMSDs, Q values, contact distances, etc.,
can generate specialized figures in the same step by passing the keyword
argument ``plot=True``.
3. User-friendly and simple application. Where possible, the provided functions
combine many individual steps into comprehensive workflows with additional
automation features. It is possible to rapidly create whole setup or
structure-analysis workflows within a few commands, thereby significantly
enhancing productivity and reducing the time spent at various stages of the
project.

With `pyrexMD`, it becomes especially easy to create, share, and reproduce
research results or transfer the work to other biomolecular structures of
interest. Furthermore, it lowers the technical boundaries for newcomers who want
to use Replica Exchange for enhanced sampling.

`pyrexMD` should be used with `jupyter` notebooks and requires `GROMACS` to run
MD simulations.


## Documentation
You can access the documentation via https://kit-mbs.github.io/pyrexMD/
<br/>and the quick guide via https://kit-mbs.github.io/pyrexMD/quick_guide.html.


## Installation Instructions
(1) Install MPI:

On ubuntu:

    sudo apt-get install openmpi-bin libopenmpi-dev

On macOS:

    brew install mpich

(2) Download and install an MPI-enabled version of `GROMACS` following the
instructions given below:

https://manual.gromacs.org/documentation/current/index.html


(3) Download and install the `pyrexMD` package:

    git clone https://github.com/KIT-MBS/pyrexMD.git
    cd pyrexMD
    python -m pip install .


(4) Enable the trajectory viewer for `jupyter`:

    jupyter nbextension enable --py widgetsnbextension
    jupyter nbextension enable --py nglview


## Tests
To run the python tests with code coverage, go into the pyrexMD folder and run:

    coverage run -m pytest
    coverage report html

Open htmlcov with any browser to see the code coverage, e.g.:

    firefox htmlcov/index.html


## Examples
To run one of the short examples, go into the pyrexMD/examples folder and run:

    jupyter notebook

## Participating
`pyrexMD` is free and open source. It is published under the MIT license. We
welcome your contribution so feel free to ask questions, report bugs, fork, etc.
