---
title: 'pyrexMD: Workflow-Orientated Python Package for Replica Exchange Molecular Dynamics'
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
control or advanced drug design. Molecular dynamics (MD) is a computational
method relying on physical models to simulate biomolecular systems. Movements of
all atoms can be 'viewed' like a movie and analyzed to improve the understanding
of specific interactions or complement experimental measurements. Replica
Exchange (REX) is a powerful method used to enhance the sampling of protein
conformations and generates large amounts of data.


`pyrexMD` is mainly designed for research projects which:

- rely on (contact-guided) REX MD or (contact-guided) MD
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
research utilizing REX for enhanced sampling.

# Theoretical background

Timescales of various biomolecular interactions of interest, such as protein
folding, conformation transitions, or ligand binding, are typically in the order
of µs to s. MD simulations, however, operate in 1-2 fs steps, which makes
in-silico studies of proteins computationally demanding. Besides, observations
of native configurations during MD simulations are often not guaranteed and
proteins can become trapped in certain conformations. One possibility to
overcome this problem is to utilize an enhanced sampling technique such as REX
[@sugita1999replica; @zhang2005convergence]. REX simulates N non-interacting
copies (“replicas”) of a system at different temperatures T$_i$. At certain time
intervals adjacent replicas can be exchanged which leads to a walk in
temperature space. REX is therefore suited to obtain physically meaningful
structure ensembles at specific temperatures. Based on the chosen temperature
range and distribution is is also possible to obtain native-like conformations
within a single run. Depending on the research goal, it is beneficial to
integrate additional theoretically [@morcos2011direct] or experimentally derived
[@perilla2017cryoem] biases into REX simulations to restrict the sampling space
and thus effectively lower computational costs.


# Statement of need

In particular analyzing research studies relying on REX can become quite arduous
and time consuming. REX simulations usually not only require knowledge of
various program tools but also consist of many individual steps ranging from
simulation setup and pre-processing over testing and simulation-monitoring to
post-processing and data analyses. Furthermore, REX can generate terabytes of
data and requires in particular a systematic handling of I/O.

One of the most used software packages for MD is `Gromacs` [@van2005gromacs],
a free open source solution enabling the user to choose from many different
force fields such as  GROMOS [@schmid2011definition], AMBER
[@wang2004development], CHARMM [@mackerell2000development], or OPLS
[@jorgensen1996development]. The core functionality of `Gromacs` can be extended
by plug-ins such as `PLUMED` [@bonomi2009plumed; @tribello2014plumed] or
`SSAGES` [@sidky2018ssages]. They implement additional algorithms and enhanced
sampling methods which interact during the MD simulation itself or can give
access to user-defined collective variables enabling new types of analyses.

`pyrexMD` on the other hand focuses on improvements during the simulation setup
or post-simulation analyses. It provides efficient and robust methods for
setting up optimized (contact-guided) REX MD or MD simulations. Furthermore it
offers many intuitive and user-friendly structure analyses and comparison
functions to explore the large I/O sets generated by REX.

Examples of currently available functions include:

- set up systems for MD or REX MD simulations
- integration of bias contacts and bias potentials
- topology comparison functions for consistency checks across different systems or replicas
- trajectory viewer and interactive plots
- wide range of functions related to structure analyses and comparison (e.g.
  contact maps/distances, RMSD, Qvalues, Global Distance Test, local accuracy,
  dihedrals, cluster analyses, etc.)
- easy creation of specialized figures for data vizualization
- automated logging functions

`pyrexMD` efficiently integrates and extends the following popular MD-related
python packages:

- `MDAnalysis` [@oliver_beckstein-proc-scipy-2016; @michaud2011mdanalysis],
- `nglview` [@nguyen2018nglview],
- `GromacsWrapper` [@oliver_beckstein_2019_2654393].

By covering various important aspects, `pyrexMD` allows to execute the whole
project from start to finish without switching to other programs which
unnecessarily interrupts the workflow and often requires know-how of different
command line syntaxes. Alongside many workflow-orientated functions, it also
adds a variety of useful general functions and quality of life improvements,
such as an integrated trajectory viewer, interactive figures linked to a
trajectory or creation of multi-panel figures from saved .pickle files to reuse
individual or old figures without requiring the explicit data set.

# Example applications

`pyrexMD` was initially developed during the work of [@voronin2020including]. It
is currently applied in ongoing REX studies about protein and RNA structure
refinement.

Fig. 1 exemplarily shows the application of the trajectory viewer with an
interactive plot. Fig. 2 displays a true positive rate analysis of predicted
bias contacts which are considered for a contact-guided REX simulation. Fig. 3
shows the local accuracy of conformations based on a Global Distance Test for
models generated during a REX study.

![Trajectory viewer on top which is linked to an interactive plot (here RMSD) at
the bottom. It is possible to quickly inspect conformations at specific values
by interacting with the graph itself (e.g. via ctrl-click) in order to get
additional valueable information accessible through the trajectory
viewer.](figs/fig1.png){ width=80% }

![Analysis of the true positive rate (TPR) for bias contacts using `pyrexMD`.
The figure exemplarily shows the TPR of considered bias contacts in addition to
other relevant value guidelines for contact-guided REX according to
[@voronin2020including] such as a minimal TPR threshold of 75% in red and a
suggested optimal number of contacts in orange between L/2 and L contacts,
vizualized by the orange region.](figs/fig2.png){ width=90% }

![Local accuracy of REX-generated protein models sorted by Global Distance Test
scores. The so-called Global Distance Test (GDT) is a method for structure
evaluation comparable to the root-mean-square deviation (RMSD). The figure gives
a clear representation of how good each model part is refined compared to a
reference structure. Each residue is color-coded to represent the CA-CA distance
between the model and reference structure after fitting. The two corresponding
GDT score variants Total Score (TS) and High Accuracy (HA) are shown on the left
side.](figs/fig3.png){ width=90% }

# Availability

`pyrexMD` can be downloaded from
[https://github.com/KIT-MBS/pyrexMD](https://github.com/KIT-MBS/pyrexMD) under the
MIT license. Both online documentation and quick guide can be accessed via
[https://kit-mbs.github.io/pyrexMD](https://kit-mbs.github.io/pyrexMD)


# Acknowledgments

We want to thank all developers and contributors of MDAnalysis as the
backbone of pyrexMD allowing us to parse and analyze MD-related data files.

This work is supported by the Helmholtz Association Initiative and Networking
Fund under project number ZT-I-0003.

# References
