Quick Guide
===========

Here you can find a short overview of some core functions and workflows which
can be performed with pyrexMD. Since pyrexMD is workflow orientated, you can
perform large tasks (e.g. system setup, specific analyses, etc.) within a few
commands. This guide is kept very short intentionally because most commands have
obvious 'core'-functionality. Plot-related functions have a wide range of
possible keyword arguments, in which case you have to dig through the API docs
to get the most potential out of the functions.


Setup of normal MD
------------------

    import pyrexMD.gmx as gmx

    # create ref pdb:
    pdb = "./1l2y.pdb"
    ref = gmx.get_ref_structure(pdb, ff='amber99sb-ildn', water='tip3p', ignh=True)

    # generate topology & box
    gmx.pdb2gmx(f=ref, o="protein.gro", ff='amber99sb-ildn', water='tip3p', ignh=True)
    gmx.editconf(f="protein.gro", o="box.gro", d=2.0, c=True, bt="cubic")

    # generate solvent & ions
    gmx.solvate(cp="box.gro", o="solvent.gro")
    gmx.grompp(f="ions.mdp", o="ions.tpr",c="solvent.gro")
    gmx.genion(s="ions.tpr", o="ions.gro", neutral=True, input="SOL")

    # minimize
    gmx.grompp(f="min.mdp", o="min.tpr", c="ions.gro")
    gmx.mdrun(deffnm="min")

    # NVT equilibration
    gmx.grompp(f="nvt.mdp", o="nvt.tpr", c="min.gro", r="min.gro")
    gmx.mdrun(deffnm="nvt")

    # NPT equilibration
    gmx.grompp(f="npt.mdp", o="npt.tpr", c="nvt.gro", r="nvt.gro", t="nvt.cpt")
    gmx.mdrun(deffnm="npt")

    # MD run
    gmx.grompp(-f"md.mdp", o="traj.tpr", c="npt.gro", t="npt.cpt")
    gmx.mdrun(deffn="traj")



Setup of contact-guided REX MD
------------------------------

    import pyrexMD.misc as misc
    import pyrexMD.rex as rex
    import pyrexMD.topology as top

    decoy_dir = "path/to/decoy/directory"

    # create rex_i directories and assign decoys
    rex.assign_best_decoys(decoy_dir)
    rex_dirs = rex.get_REX_DIRS()

    # check for consistent topology
    rex.check_REX_PDBS(decoy_dir)

    # dump mdp files (min.mdp, nvt.mdp, npt.mdp, rex.mdp)
    misc.cp("path/to/mdp/files", ".")

    # get parameters for fixed boxsize and solvent molecules
    boxsize, maxsol = rex.WF_get_system_parameters(wdir="./rex_0_get_system_parameters/")

    # create systems for each replica
    rex.WF_REX_setup(rex_dirs=rex_dirs, boxsize=boxsize, maxsol=maxsol)

    # minimize
    rex.WF_REX_setup_energy_minimization(rex_dirs=rex_dirs, nsteps=10, verbose=False)

    # add bias contacts (defined in DCA_fin)
    top.DCA_res2atom_mapping(ref_pdb=<ref_pdb>, DCA_fin=<file_path>, n_DCA=50, usecols=(0,1))
    top.DCA_modify_topology(top_fin="topol.top", DCA_used_fin=<file_path> , k=10, save_as="topol_mod.top")

    # prepare temperature distribution
    rex.prep_REX_temps(T_0=280, n_REX=len(rex_dirs), k=0.006)

    # create mdp and tpr files
    rex.prep_REX_mdp(main_dir="./", n_REX=len(rex_dirs))
    rex.prep_REX_tpr(main_dir="./", n_REX=len(rex_dirs))

    # next: upload run files on HPC and execute production run



Interactive Plots
--------------------


    import MDAnalysis as mda
    import pyrexMD.misc as misc
    import pyrexMD.core as core
    import pyrexMD.topology as top
    import pyrexMD.analysis.contacts as con


    ref = mda.Universe("<pdb_file>")
    mobile = mda.Universe("<tpr_file>", "<xtc_file>")
    top.norm_and_align_universe(mobile, ref)

    # show ref structure in trajectory viewer
    tv = core.iPlayer(ref)
    tv()

    # check RMSD
    FRAMES, TIME, RMSD = ana.get_RMSD(mobile, ref, sel1="protein", sel2="protein", plot=True)

    # check for formed bias contacts
    bias = misc.read_file("path/to/bias/contacts", usecols=(0,1))
    FRAMES, QBIAS, CM = con.get_QBias(mobile, bias)

    # interactive plot (ctrl-click into the plot to jump to frame)
    ip = core.iPlot(u1, xdata=FRAMES, ydata=RMSD, xlabel="frame", ylabel=r"RMSD (A)")
    ip()



Contact and Bias Analyses
--------------------------

    TEST NEW CHANGE2


GDT and LA Analyses
-------------------
    import MDAnalysis as mda
    import pyrexMD.misc as misc
    import pyrexMD.core as core
    import pyrexMD.topology as top
    import pyrexMD.analysis.analysis as ana
    import pyrexMD.analysis.gdt as gdt

    ref = mda.Universe("<pdb_file>")
    mobile = mda.Universe("<tpr_file>", "<xtc_file>")
    top.norm_and_align_universe(mobile, ref)

    # perform GDT (Global Distance Test)
    GDT = gdt.GDT(mobile, ref)
    GDT_percent, GDT_resids, GDT_cutoff, RMSD, FRAME = GDT

    # calculate GDT scores
    GDT_TS = gdt.get_GDT_TS(GDT_percent)
    GDT_HA = gdt.get_GDT_HA(GDT_percent)

    # rank scores
    SCORES = gdt.GDT_rank_scores(GDT_percent, ranking_order="GDT_TS", verbose=False)
    GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked = SCORES

    # generate plots
    ana.PLOT(xdata=frames, ydata=GDT_TS, xlabel="Frame", ylabel="GDT TS")
    ana.plot_hist(GDT_TS, n_bins=20, xlabel="GDT TS", ylabel="Counts")

    # Local Accuracy plot
    gdt.plot_LA(mobile, ref, GDT_TS_ranked, GDT_HA_ranked, GDT_ndx_ranked,



Cluster Analyses
----------------

    import pyrexMD.misc as misc
    import pyrexMD.decoy.cluster as clu

    # load data
    QDATA = misc.pickle_load("./data/QDATA.pickle")
    RMSD = misc.pickle_load("./data/RMSD.pickle")
    GDT_TS = misc.pickle_load("./data/GDT_TS.pickle")
    score_file = "./data/energies.log"
    ENERGY = misc.read_file(score_file, usecols=1, skiprows=1)
    DM = clu.read_h5("./data/DM.h5")

    # apply TSNE for dimension reduction
    tsne = clu.apply_TSNE(DM, n_components=2, perplexity=50, random_state=1)

    ### apply KMeans on TSNE-transformed data (two variants with low and high cluster number)
    # note: here we set the high number only to 20 because our sample is small with only 500 frames
    cluster10 = clu.apply_KMEANS(tsne, n_clusters=10, random_state=1)
    cluster20 = clu.apply_KMEANS(tsne, n_clusters=20, random_state=1)

    ### map scores (energies) and accuracy (GDT, RMSD) to clusters
    cluster10_scores = clu.map_cluster_scores(cluster_data=cluster10, score_file=score_file)
    cluster10_accuracy = clu.map_cluster_accuracy(cluster_data=cluster10, GDT=GDT_TS, RMSD=RMSD)
    cluster20_scores = clu.map_cluster_scores(cluster_data=cluster20, score_file=score_file)
    cluster20_accuracy = clu.map_cluster_accuracy(cluster_data=cluster20, GDT=GDT_TS, RMSD=RMSD)

    ### plot cluster data
    # here: TSNE-transformed data with n_clusters = 10
    # also: plot cluster centers with different colors
    #     - red dot: n20 centers
    #     - black dot: n10 centers
    clu.plot_cluster_data(cluster10, tsne, ms=40)
    clu.plot_cluster_center(cluster10, marker="o", color="red", ms=20)
    clu.plot_cluster_center(cluster20, marker="o", color="black")

    ### plot cluster data
    # here: TSNE-transformed data with n_clusters = 20
    # also: plot cluster centers with different colors
    #     - red dot: n20 centers
    #     - black dot: n10 centers
    clu.plot_cluster_data(cluster20, tsne)
    clu.plot_cluster_center(cluster10, marker="o", color="red", ms=20)
    clu.plot_cluster_center(cluster20, marker="o", color="black")

    ### print table with cluster scores stats
    clu.WF_print_cluster_scores(cluster_data=cluster10, cluster_scores=cluster10_scores)
    clu.WF_print_cluster_scores(cluster_data=cluster20, cluster_scores=cluster20_scores)

    ### print table with cluster accuracy stats
    clu.WF_print_cluster_accuracy(cluster_data=cluster10, cluster_accuracy=cluster10_accuracy)
    clu.WF_print_cluster_accuracy(cluster_data=cluster20, cluster_accuracy=cluster20_accuracy)
