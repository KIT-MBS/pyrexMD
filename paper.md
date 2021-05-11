---
title: 'pyREX: Workflow-Oriented Python Package for Replica-Exchange Molecular Dynamics'
tags:
  - Python
  - Gromacs
  - physics
  - biology
  - replica exchange
  - molecular dynamics
  - contact bias
authors:
  - name: Arthur Voronin
    orcid: 0000-0002-5804-5151
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Alexander Schug^[*]
    orcid: 0000-0002-0534-502X
    affiliation: "3, 4" # (Multiple affiliations must be quoted)
affiliations:
 - name: Steinbuch Centre for Computing, Karlsruhe Institute of Technology, Eggenstein-Leopoldshafen, Germany
   index: 1
 - name: Department of Physics, Karlsruhe Institute of Technology, Karlsruhe, Germany
   index: 2
 - name: Institute for Advanced Simulation, Jülich Supercomputing Center, Jülich, Germany
   index: 3
 - name: Faculty of Biology, University of Duisburg-Essen, Duisburg, Germany
   index: 4
date: 10 May 2021
bibliography: paper.bib

---

# Summary

Proteins are complex biomolecules which fulfil important and diverse tasks in
living organism. Studying and understanding their structure, function and
dynamics is essential for life sciences and can, for instance, be applied for
disease control or to improve drug design. Molecular dynamics (MD) is a
computational method which relies on phyisical models to simulate biomolecular
systems. The movements of all atoms can be "viewed" like a movie and analyzed to
improve the understanding of specific interactions or complement conducted
experiments. Replica-Exchange (REX [@sugita1999replica:1991,
@zhang2005convergence:2005]) is a powerful method used to enhance sampling of
protein conformations which generates large amounts of data. `pyREX` is designed
as an interactive "all-purpose" toolkit for research projects which rely on
(contact-guided) Replica-Exchange Molecular Dynamics using GROMACS
[@van2005gromacs:2005]. Its workflow-orientated design and existing templates
significantly enhance productivity and reduce the time spend at various stages
of the project.


 - ref to PLOS ONE paper ?

# Statement of need


Timescales of various biomolecular interactions of interest, such as protein
folding, conformation transitions or ligand bonding, are typically in the order
of µs to s. MD simulations, however, operate in 1-2 fs steps, which makes
in-silico studies of proteins computational demanding. Besides, observations of
native state configurations during MD simulations are often not guaranteed and
proteins can get trapped in certain conformations. One possibility to overcome
this problem is to utilize one of the many enhanced sampling techiques, such as
REX. Depending on the research goal it is also possible to integrate additional
theoretically-derived [@morcos2011direct:2011] or experimentally-derived
[@perilla2017cryoem:2017] bias into MD simulations and reduce the search space to
effectively lower the computational costs.

Research studies relying on REX, however, can become very ardous and time
consuming. REX simulations usually require not only knowledge of various program
tools but also consist of many individual steps ranging from simulation setup,
pre-processing, testing, monitoring the simulations, post-processing to data
analyses. Furthermore, REX generates large amounts of data and requires a good
structure when handling I/O.

`pyREX` is designed as an interactive "all-purpose" toolkit specifically for
(contact-guided) REX projects using GROMACS and is meant to be used in
jupyter notebooks. It integrates some popular MD-related python packages, e.g.:
- MDAnalysis [@gowers2019mdanalysis:2019 ; @michaud2011mdanalysis:2011],
- nglview [@nguyen2018nglview:2018],
- GromacsWrapper [@oliver_beckstein_2019_2654393:2019].

By covering all important aspects, `pyREX` eliminates the need of switching to
additional programs which unnecesary interrupts the workflow and often requires
know-how of different command line syntaxes. Furthermore, 'pyREX' offers many
workflow-orientated functions and templates for different stages of a REX study.
Its main focuses are 1) to provide efficient and robust methods for setting up
optimized (contact-guided) REX simulations and 2) to offer many different
structure analyses functions to exploit the large I/O sets generated by REX.
It also adds a variety of useful general functions and QoL improvements, such as
interactively linked figures and trajectory or creation of multi-panel figures
from .pickle files. With `pyREX` it becomes especially easy to share and
reproduce research results or transfer the work on other target structures of
interest. Furthermore, it lowers the technical boundaries for newcomers who want
to do research using REX for enhanced sampling.

# show some figures
-rex setup

-global rmsd heatmap

-local accuracy


# Citations


# Acknowledgements

We want to thank all developers and contributors of MDAnalysis, which is the
backbone of pyREX as it allows us to parse and analyze MD-related data files.


This work is supported by the Helmholtz Association Initiative and Networking
Fund under project number ZT-I-0003.

# References