[![DOCS](https://img.shields.io/badge/docs-1.0-blue?&logo=github)](https://kit-mbs.github.io/pyrexMD/)
[![GH Actions CI](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml/badge.svg)](https://github.com/KIT-MBS/pyrexMD/actions/workflows/gh-actions-ci.yml)
[![codecov](https://codecov.io/gh/KIT-MBS/pyrexMD/branch/master/graph/badge.svg?token=TdmhhPgQNW)](https://codecov.io/gh/KIT-MBS/pyrexMD)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.03325/status.svg)](https://doi.org/10.21105/joss.03325)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5744760.svg)](https://doi.org/10.5281/zenodo.5744760)

About pyrexMD
=============

`pyrexMD` is a Python package that is mainly designed for research projects which

- use (contact-guided) Replica Exchange Molecular Dynamics or (contact-guided) Molecular Dynamics
- or focus on structure analyses and comparison.

It has three main goals:

1. Interactive 'all-purpose' environment. By including various modified
`GROMACS` and `MDAnalysis` Python bindings, this package provides a
comprehensive `Jupyter` notebooks based environment to design, run, and analyze
MD simulation projects from beginning to end.
2. Data visualization is important. In `pyrexMD`, most analysis functions for
calculating useful quantities, such as root-mean-square deviation (RMSD), Q
values, contact distances, etc., can generate specialized figures in the same
step by passing the keyword argument ``plot=True``.
3. User-friendly and simple application. Where possible, the provided functions
combine individual steps into comprehensive workflows with additional automation
features. It is possible to rapidly create whole setup or structure-analysis
workflows within a few commands, thereby significantly enhancing productivity
and reducing the time spent on various stages of the project.

`pyrexMD` makes it straightforward to create, share, and reproduce
research results or transfer the work to other biomolecular structures of
interest. Furthermore, it lowers the technical barrier for non-specialists who
want to use Replica Exchange for enhanced sampling.

`pyrexMD` should be used with `Jupyter` notebooks and requires `GROMACS` to run
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


(4) Enable the trajectory viewer for `Jupyter`:

    jupyter nbextension enable --py widgetsnbextension
    jupyter nbextension enable --py nglview


## Tests
To run the Python tests with code coverage, go into the pyrexMD folder and run:

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


## Citation

Voronin et al., (2021). pyrexMD: Workflow-Orientated Python Package for Replica Exchange Molecular Dynamics. Journal of Open Source Software, 6(68), 3325, https://doi.org/10.21105/joss.03325

bibtex entry:

    @article{Voronin2021,
    doi = {10.21105/joss.03325},
    url = {https://doi.org/10.21105/joss.03325},
    year = {2021},
    publisher = {The Open Journal},
    volume = {6},
    number = {68},
    pages = {3325},
    author = {Arthur Voronin and Alexander Schug},
    title = {{pyrexMD: Workflow-Orientated Python Package for Replica Exchange Molecular Dynamics}},
    journal = {Journal of Open Source Software}
    }

## License

MIT License

Copyright (c) 2021 KIT-MBS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
