from __future__ import division, print_function
from tqdm import tqdm
import my.misc as _misc
import my.analysis as _ana
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing
import MDAnalysis as mda
import pyrosetta
from pyrosetta.rosetta import protocols
pyrosetta.init("-mute core.import_pose core.pack.task core.pack.pack_rotamers protocols.relax.FastRelax core.pack.interaction_graph.interaction_graph_factory")  # mute messages
#pyrosetta.init("-mute core.import_pose")  # mute all import pose messages


def abinitio_setup_cfg(pdbid, fasta_seq, frag3mer, frag9mer, **kwargs):
    """
    Setup config file for abinitio functions.

    Args:
        pdbid (str): pdb id / pdb name
        fasta_seq (str): fasta sequence
        frag3mer (str): 3mer file path (get via robetta server)
        frag9mer (str): 9mer file path (get via robetta server)

    Kwargs:
        fasta_seq_len (int): fasta sequence length
        frag3inserts (int): number of frag3 inserts
        frag9inserts (int): number of frag9 inserts
        folding_cycles (int): folding move cycles
        folding_repeats (int): folding move repeats
        job_cores (int): -np option for multiprocessing
        job_repeats (int): total number of job repeats
        job_name (str)
        decoy_ndx_shift (int): shift decoy index (output filename) by this value
        kT (float): kT parameter during Monte-Carlo simulation

    Returns:
        cfg (CONFIG class)
    """
    default = {"pdbid": pdbid,
               "fasta_seq": fasta_seq,
               "fasta_seq_len": None,
               "frag3mer": frag3mer,
               "frag9mer": frag9mer,
               "frag3inserts": 3,
               "frag9inserts": 1,
               "folding_cycles": 1000,
               "folding_repeats": 10,
               "job_cores": 10,
               "job_repeats": 10,
               "job_name": pdbid,
               "decoy_ndx_shift": 0,
               "kT": 1.0}
    cfg = _misc.CONFIG(default)
    cfg.update_config(fasta_seq_len=len(cfg.fasta_seq))
    cfg.update_config(**kwargs)
    return cfg


def abinitio_create_decoys(abinitio_cfg, output_dir="./output",
                           stream2pymol=True, fastrelax=True, save_log=True):
    """
    Args:
        abinitio_cfg (CONFIG class)
        output_dir (str): output directory for decoys
        stream2pymol (bool): stream decoys to pymol
        fastrelax (bool): apply fastrelax protocol on decoys before dumping them as pdb
        save_log (bool): save scores to logfile at <output_dir/scores.txt>
    """
    cfg = abinitio_cfg

    ### create output directory
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    if os.path.exists(output_dir):
        msg = f"""Output directory '{output_dir}' already exists.
Creating decoys will overwrite existing decoys in this directory.
Proceed? [y/n]"""
        answer = input(msg).lower()
        if (answer == "y" or answer == "yes"):
            pass
        if (answer == "n" or answer == "no"):
            return
    _misc.mkdir(output_dir)

    ### create decoys code
    # conversion movers
    to_centroid = pyrosetta.SwitchResidueTypeSetMover('centroid')
    to_fullatom = pyrosetta.SwitchResidueTypeSetMover('fa_standard')

    # score function and score array
    scorefxn_low = pyrosetta.create_score_function('score3')
    scorefxn_high = pyrosetta.create_score_function('ref2015')
    #scorefxn_high = get_fa_scorefxn() # ref2015

    SCORES_low = [0]*(cfg.job_repeats)  # scores array
    SCORES_high = [0]*(cfg.job_repeats)  # scores array

    ### pose objects
    # linear pose
    pose_0 = pyrosetta.pose_from_sequence(cfg.fasta_seq)
    pose_0.pdb_info().name(f"{cfg.pdbid} (linear)")

    # test pose
    pose = pyrosetta.Pose()
    pose.assign(pose_0)
    pose.pdb_info().name(cfg.pdbid)

    # switch to centroid
    to_centroid.apply(pose_0)
    to_centroid.apply(pose)

    ### mover and fragset objects
    movemap = pyrosetta.MoveMap()
    movemap.set_bb(True)

    fragset_3mer = pyrosetta.rosetta.core.fragment.ConstantLengthFragSet(3, cfg.frag3mer)
    fragset_9mer = pyrosetta.rosetta.core.fragment.ConstantLengthFragSet(9, cfg.frag9mer)

    mover_frag3 = protocols.simple_moves.ClassicFragmentMover(fragset_3mer, movemap)
    mover_frag9 = protocols.simple_moves.ClassicFragmentMover(fragset_9mer, movemap)

    insert_frag3 = protocols.moves.RepeatMover(mover_frag3, cfg.frag3inserts)
    insert_frag9 = protocols.moves.RepeatMover(mover_frag9, cfg.frag9inserts)

    folding_mover = protocols.moves.SequenceMover()
    folding_mover.add_mover(insert_frag9)
    folding_mover.add_mover(insert_frag3)

    # MC stuff
    mc = pyrosetta.MonteCarlo(pose, scorefxn_low, cfg.kT)
    trial = pyrosetta.TrialMover(folding_mover, mc)
    folding = protocols.moves.RepeatMover(trial, cfg.folding_cycles)
    #jd = PyJobDistributor(cfg.job_name, cfg.job_repeats, scorefxn_high)

    if stream2pymol:
        pmm = pyrosetta.PyMOLMover()
        pmm.keep_history(True)

    ### job distributor stuff
    for i in range(cfg.job_repeats):
        print(f"\n>>> Working on decoy: {output_dir}/{cfg.job_name}_{cfg.decoy_ndx_shift+i}.pdb")
        pose.assign(pose_0)
        pose.pdb_info().name(f"{cfg.job_name}_{cfg.decoy_ndx_shift+i}")
        mc.reset(pose)
        for j in range(cfg.folding_repeats):
            folding.apply(pose)
            mc.recover_low(pose)
        SCORES_low[i] = scorefxn_low(pose)
        to_fullatom.apply(pose)

        if fastrelax:
            relax = protocols.relax.FastRelax()
            relax.set_scorefxn(scorefxn_high)
            relax.apply(pose)

        SCORES_high[i] = scorefxn_high(pose)
        pose.dump_pdb(f"{output_dir}/{cfg.job_name}_{cfg.decoy_ndx_shift+i}.pdb")

        if stream2pymol:
            pmm.apply(pose)

    if save_log:
        logfile = f"{output_dir}/scores.txt"
        write_header = True
        with open(logfile, "r") as log:
            if "ref2015" in log.readline():
                write_header = False
        with open(logfile, "a") as log:
            if write_header:
                log.write(f"{'decoy'}\t{'score3'}\t{'ref2015'}\n")
            DECOYS = [f"{cfg.job_name}_{cfg.decoy_ndx_shift+i}" for i in range(cfg.job_repeats)]
            table_str = _misc.print_table([DECOYS, SCORES_low, SCORES_high])
            log.write(table_str)

    return SCORES_low, SCORES_high


def abinitio_align_universe(decoy, ref, print_comment=False):
    """
    ref protein is smaller than decoy:
        -> shift resids and atomids of ref to match decoy.

    Args:
        decoy (MDA universe): pyrosetta decoy pdb path
        ref (MDA universe): reference structure pdb path
        print_comment (bool): print comment about this function's usage

    Returns:
        decoy (MDA atomgrp)
    """
    decoy = _ana.true_select_atoms(u=decoy, sel='protein', ignh=True)
    decoy.atoms.ids = _misc.norm_array(decoy.atoms.ids, start_value=decoy.atoms.ids[0])
    # get slice_indices by comparing resnames
    slice_ndx1, slice_ndx2 = _misc.get_slice_indices(ref.residues.resnames, decoy.residues.resnames)
    ref.residues.resids = decoy.residues.resids[slice_ndx1:slice_ndx2]

    # get atom id shift by counting atoms with resid < (new) smallest ref resid
    atom_id_shift = len(np.where(decoy.atoms.resids < ref.atoms.resids[0])[0])
    ref.atoms.ids += atom_id_shift

    if print_comment:
        print("The function 'abinitio_align_universe' returns a decoy atom_grp!")
        print("This does not affect the original decoy universe!")

    return decoy


def abinitio_select_RMSD_atomgrp(decoy, ref):
    """
    Args:
        decoy (MDA atomgrp)
        ref (MDA universe)

    Returns:
        decoy (MDA atomgrp)
    """
    resmin = ref.residues.resids[0]
    resmax = ref.residues.resids[-1]
    decoy = _ana.true_select_atoms(decoy, sel=f"protein and resid {resmin}-{resmax}")
    return decoy


def _HELP_str2int(list_):
    """
    HELP FUNCTION: convert list with strings to list with ints

    Example:
        L = ['./output/2hda_4444.pdb', './output/2hda_4852.pdb',
             './output/2hda_1813.pdb', './output/2hda_3177.pdb']
        _HELP_str2int(L)
        >> [4444, 4852, 1813, 3177]
    """

    ints_ = []
    for str_ in list_:
        if isinstance(str_, (int, np.integer, np.signedinteger)):
            ints_.append(str_)
        elif isinstance(str_, str):
            substrings = _misc.get_substrings(str_)
            for sub in substrings:
                try:
                    ints_.append(int(sub))
                    break  # stop substring iteration if integer is found
                except ValueError:
                    pass
        else:
            print("abinitio._HELP_str2int(list_):\nWarning:")
            print(f"{str_} is item in <list_> and is not of integer- or string-type.")
    return ints_


def abinitio_get_decoy_precission(ref, decoy_prefix="./output/2hda_", decoy_ndx_range=(None, None),
                                  return_decoy_ndx=True):
    """
    get arrays with
        - Decoy(names)
        - Score
        - RMSD
    to compare decoy precission.

    Args:
        ref (MDA universe): reference structure
        decoy_prefix (str): prefix of decoy path.
                            Example:
                                decoy path: './output/decoy_0.pdb'
                                decoy prefix: './output/decoy_'
        decoy_ndx_range (tuple/list):
            use only decoy with indices within [ndx_start, index_stop].
            if (None, None): use all decoys found in decoy_prefix folder
        return_decoy_ndx (bool):
            True: DECOY array contains only decoy indices
            False: DECOY array contains the full decoy names/paths

    Returns:
        DECOY (list): list with decoy names
        SCORE (list): list with corresponding scores (ref 2015)
        RMSD (list): list with corresponding RMSD values
    """
    decoy_files = glob.glob(f"{decoy_prefix}*")

    # get decoy index range if not passed
    decoy_ndx_range = list(decoy_ndx_range)  # make sure that new list object is created
    search = False
    if decoy_ndx_range[0] is None:
        decoy_ndx_range[0] = 9999999999  # min_ndx
        search = True
    if decoy_ndx_range[1] is None:
        decoy_ndx_range[1] = 0           # max_ndx
        search = True
    if search:
        for file in decoy_files:
            substrings = _misc.get_substrings(file)
            for sub in substrings:
                try:
                    integer = int(sub)
                    if integer < decoy_ndx_range[0]:
                        decoy_ndx_range[0] = integer
                    if integer > decoy_ndx_range[1]:
                        decoy_ndx_range[1] = integer
                except ValueError:
                    pass

    scorefxn = pyrosetta.get_fa_scorefxn()
    DECOY = []  # list with decoy names
    SCORE = []  # list with corresponding scores
    RMSD = []   # list with corresponding RMSD

    for i in tqdm(range(decoy_ndx_range[0], decoy_ndx_range[1]+1)):
        decoy_path = f"{decoy_prefix}{i}.pdb"
        if not os.path.exists(decoy_path):
            print(f"The decoy path '{decoy_path}' does not exist!")
            break

        # set DECOY
        DECOY.append(decoy_path)

        # get SCORE
        pose = pyrosetta.pose_from_pdb(decoy_path)
        SCORE.append(scorefxn(pose))

        # get RMSD
        decoy = mda.Universe(decoy_path)
        decoy = abinitio_align_universe(decoy, ref, print_comment=False)
        decoy_RMSD_atomgrp = abinitio_select_RMSD_atomgrp(decoy, ref)
        ftr = _ana.get_RMSD(decoy_RMSD_atomgrp, ref, sel='backbone')
        RMSD.append(ftr[2])

    RMSD = _misc.flatten_array(RMSD)

    if return_decoy_ndx:
        DECOY = _HELP_str2int(DECOY)
    return DECOY, SCORE, RMSD


def abinitio_rank_decoy_arrays(DECOY, SCORE, RMSD, rank_by="SCORE", return_decoy_ndx=True):
    """
    'Link' input data and then rank based on score in ascending order (low to high).
    Input data is output of abinitio_get_decoy_precission() function.

    Args:
        DECOY (list/array): array with decoy names
        SCORE (list/array): array with score
        RMSD (list/array): array with RMSD
        rank_by (str): "SCORE", "RMSD"
        return_decoy_ndx (bool):
            True: DECOY_ranked array contains only decoy indices
            False: DECOY_ranked array contains the full decoy names

    Return:
        DECOY_ranked (array): ranked array
        SCORE_ranked (array): ranked array
        RMSD_ranked (array): ranked array
    """
    if rank_by.upper() == "SCORE":
        SCORE_ranked, SCORE_ranked_ndx = _misc.get_ranked_array(SCORE, reverse=True)
        DECOY_ranked = np.array([DECOY[ndx] for ndx in SCORE_ranked_ndx])
        RMSD_ranked = np.array([RMSD[ndx] for ndx in SCORE_ranked_ndx])
    if rank_by.upper() == "RMSD":
        RMSD_ranked, RMSD_ranked_ndx = _misc.get_ranked_array(RMSD, reverse=True)
        DECOY_ranked = np.array([DECOY[ndx] for ndx in RMSD_ranked_ndx])
        SCORE_ranked = np.array([SCORE[ndx] for ndx in RMSD_ranked_ndx])

    if return_decoy_ndx:
        DECOY_ranked = _HELP_str2int(DECOY_ranked)
    return DECOY_ranked, SCORE_ranked, RMSD_ranked


def abinitio_scatterplot(SCORE, RMSD, **kwargs):
    """
    Args:
        SCORE (list/array)
        RMSD (list/array)

    Kwargs:
        # see args of misc.figure()

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
    """

    fig, ax = _misc.figure(**kwargs)
    plt.plot(RMSD, SCORE, ".")
    plt.xlabel(r"RMSD ($\AA$)", fontweight="bold")
    plt.ylabel("SCORE (REU)", fontweight="bold")
    return fig, ax
