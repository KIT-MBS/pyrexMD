---
title: 'pyrexMD: Workflow-Orientated Python Package for Replica Exchange Molecular Dynamics'
tags:
  - Python
  - GROMACS
  - physics
  - biology
  - replica exchange
  - molecular dynamics
  - contact bias
authors:
  - name: Arthur Voronin
    orcid: 0000-0002-5804-5151
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Alexander Schug^[al.schug\@fz-juelich.de]
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

Proteins are complex biomolecules which fulfill a wide range of critical tasks
in living organisms. Studying and understanding their structure, function, and
dynamics is essential for life sciences and can be applied for, e.g., disease
control or advanced drug design. Molecular Dynamics (MD) is a computational
method relying on physical models to simulate biomolecular systems. Movements of
all atoms can be 'viewed' like a movie and analyzed to improve the understanding
of specific interactions or complement experimental measurements. Replica
Exchange (REX) is a powerful method used to enhance the sampling of protein
conformations and generates large amounts of data.

`pyrexMD` is a Python package that is mainly designed for research projects which

- use (contact-guided) REX MD or (contact-guided) MD
- or focus on structure analyses and comparison.

It has three main goals:

1. Interactive 'all-purpose' environment. By including various modified
`GROMACS` and `MDAnalysis` Python bindings, this package provides a
comprehensive `jupyter`-notebooks based environment to design, run, and analyze
MD simulation projects from beginning to end.
2. Data visualization is important. In `pyrexMD`, most analysis functions for
calculating useful quantities, such as root-mean-square deviation (RMSD), Q
values, contact distances, etc., can generate specialized figures in the same
step by passing the keyword argument ``plot=True``.
3. User-friendly and simple application. Where possible, the provided functions
combine individual steps into comprehensive workflows with additional automation
features. It is possible to rapidly create whole setup or structure-analysis
workflows within a few commands, thereby significantly enhancing productivity
and reducing the time spent at various stages of the project.

With `pyrexMD`, it becomes straightforward to create, share, and reproduce
research results or transfer the work to other biomolecular structures of
interest. Furthermore, it lowers the technical barrier for non-specialists who
want to use REX for enhanced sampling.

# Theoretical background

Biomolecular function is often accompanied by slow conformational transitions,
typically in the order of µs to s. MD simulations, however, integrate over time
in 1-2 fs steps, which makes in-silico studies of proteins computationally
demanding. This can lead to incomplete sampling of conformational space as,
e.g., proteins can become trapped in specific conformations. One possibility to
overcome this problem is to use enhanced-sampling techniques such as REX
[@sugita1999replica; @zhang2005convergence]. REX simulates N non-interacting
copies (“replicas”) of a system at different temperatures T$_i$. After
predefined time intervals, adjacent replicas can be exchanged which leads to a
walk in temperature space, speeding up sampling while conserving thermodynamic
properties. REX is therefore suited to obtain physically meaningful ensembles of
a biomolecular structure at specific temperatures. Based on the chosen
temperature range and distribution, native-like conformations can be obtained
within a single run. Depending on the research goal, it is beneficial to
integrate additional theoretically [@schug2009high] or experimentally derived
[@perilla2017cryoem] biases into REX simulations to restrict the sampling space
and thus effectively lower computational costs.


# Statement of need

Analyzing simulation studies using REX manually is extremely arduous and
time-consuming. REX simulations usually not only require knowledge of various
program tools but also consist of many individual steps, ranging from simulation
setup and pre-processing over testing and simulation-monitoring to
post-processing and data analyses. Furthermore, REX can generate terabytes of
data and requires a systematic handling of I/O.

One of the most used software packages for MD is `GROMACS` [@van2005gromacs], a
free open-source solution providing many different force fields, such as GROMOS
[@schmid2011definition], AMBER [@wang2004development], CHARMM
[@mackerell2000development], or OPLS [@jorgensen1996development]. The core
functionality of `GROMACS` can be extended by plug-ins, such as `PLUMED`
[@bonomi2009plumed; @tribello2014plumed] or `SSAGES` [@sidky2018ssages]. Such
plug-ins implement additional algorithms and enhanced-sampling methods which
interact during the MD simulation itself or can give access to user-defined
collective variables for new types of analyses.

`pyrexMD` on the other hand focuses on facilitating, assisting, and automating
the simulation setup and post-simulation analyses. It provides efficient and
robust methods for setting up optimized (contact-guided) REX MD or MD
simulations. Furthermore, it offers many intuitive and user-friendly
structure analyses and comparison functions to explore the large I/O sets
generated by REX.

Examples of currently available functions include:

- setup of systems for MD or REX MD simulations
- integration of bias contacts and bias potentials
- topology comparison functions for consistency checks across different systems or replicas
- trajectory viewer and interactive plots
- wide range of functions related to structure analyses and comparison (e.g.
  contact maps/distances, RMSD, Q values, global distance test, local accuracy,
  dihedrals, cluster analyses, etc.)
- easy and interactive data visualization
- automation features and default-parameter switches

`pyrexMD` efficiently integrates and extends the following popular MD-related
Python packages:

- `MDAnalysis` [@oliver_beckstein-proc-scipy-2016; @michaud2011mdanalysis],
- `GromacsWrapper` [@oliver_beckstein_2019_2654393],
- `nglview` [@nguyen2018nglview].

By covering various important aspects, `pyrexMD` allows to execute the whole
project from beginning to end without switching to other programs which
unnecessarily interrupts the workflow and often requires know-how of different
command-line syntaxes. Alongside many workflow-orientated functions, it also
adds a variety of useful general functions and workload-reducing improvements,
such as an integrated trajectory viewer, interactive figures linked to a
trajectory or generation of multi-panel figures from saved .pickle files to
reuse individual or old figures without requiring the explicit data set.

# Example applications

`pyrexMD` was initially developed in the course of [@voronin2020including].
Currently, it is successfully applied in ongoing REX studies on protein and RNA
structure refinement.

Figs. 1-3 exemplarily show a small selection of possible data visualizations
after performing analyses with `pyrexMD`. Fig. 1 displays the application of the
trajectory viewer with an interactive plot. Fig. 2 shows a true-positive-rate
analysis of predicted bias contacts which are considered for a contact-guided
REX simulation. Fig. 3 visualizes the local accuracy of conformations based on a
global distance test for models obtained from a REX study.

![Trajectory viewer (top) which is linked to an interactive plot (here RMSD,
bottom). Conformations at specific values can be quickly inspected by
interacting with the graph itself (e.g. via ctrl-click), thus making additional
valuable information accessible through the trajectory viewer.](figs/fig1.png){
width=80% }

![Analysis of the true positive rate (TPR) for bias contacts with `pyrexMD`. The
figure exemplarily shows the TPR of the considered bias contacts together with
other relevant value guidelines for contact-guided REX [@voronin2020including],
such as a minimal TPR threshold of 75% (red) and a suggested optimal number of
contacts between L/2 and L (orange), where L denotes the biomolecular sequence
length.](figs/fig2.png){ width=90% }

![Local accuracy of REX-generated protein models sorted by GDT scores. The
so-called global distance test (GDT) is a method for structure evaluation
similar to the root-mean-square deviation (RMSD). This figure clearly shows how
good each model part is refined compared to a reference structure. Each residue
is color-coded to represent the CA-CA distance between the model and reference
structure after fitting. The two corresponding GDT score variants Total Score
(TS) and High Accuracy (HA) are shown on the left side.](figs/fig3.png){
width=90% }


# Availability

`pyrexMD` is free and open source. It is published under the MIT license. You
can download the package at
[https://github.com/KIT-MBS/pyrexMD](https://github.com/KIT-MBS/pyrexMD). Both
online documentation and quick guide can be accessed via
[https://kit-mbs.github.io/pyrexMD](https://kit-mbs.github.io/pyrexMD)


# Acknowledgments

We want to thank all developers and contributors of `MDAnalysis` as the
backbone of `pyrexMD` allowing us to parse and analyze MD-related data files.

This work is supported by the Helmholtz Association Initiative and Networking
Fund under project number ZT-I-0003. The authors gratefully acknowledge the
Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this
project by providing computing time through the John von Neumann Institute for
Computing (NIC) on the GCS Supercomputer JUWELS at Jülich Supercomputing Centre
(JSC). This work was also performed on the HoreKa supercomputer funded by the
Ministry of Science, Research and the Arts Baden-Württemberg and by the Federal
Ministry of Education and Research.


# References
