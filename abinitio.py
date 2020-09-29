from __future__ import division, print_function
from tqdm import tqdm_notebook as tqdm
import myPKG.misc as _misc
import myPKG.analysis as _ana
from myPKG.analysis import get_decoy_list
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
import MDAnalysis as mda
import pyrosetta
from pyrosetta.rosetta import protocols
pyrosetta.init("-mute core.import_pose core.pack.task core.pack.pack_rotamers protocols.relax.FastRelax core.pack.interaction_graph.interaction_graph_factory")  # mute messages


def setup_abinitio_cfg(pdbid, fasta_seq, frag3mer, frag9mer, **kwargs):
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
        job_name (str)
        n_decoys (int): total number of decoys
        n_cores (int): -np option for multiprocessing
        decoy_ndx_shift (int): shift decoy index (output filename) by this value
                               required for multiprocessing to fix names of decoys
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
               "job_name": pdbid,
               "n_decoys": 10,
               "n_cores": 1,
               "decoy_ndx_shift": 0,
               "kT": 1.0}
    cfg = _misc.CONFIG(default)
    cfg.update_config(fasta_seq_len=len(cfg.fasta_seq))
    cfg.update_config(**kwargs)
    return cfg


def create_decoys(abinitio_cfg, output_dir="./output", n_cores=10,
                  stream2pymol=True, fastrelax=True, save_log=True):
    """
    Create decoys within pyrosetta framework.

    Args:
        abinitio_cfg (CONFIG class): output of abinitio.setup_abinitio_cfg()
        output_dir (str): output directory for decoys
        n_cores (int): -np option for multiprocessing. Overwrites parameter
                       in abinitio_cfg
        stream2pymol (bool): stream decoys to pymol
        fastrelax (bool): apply fastrelax protocol on decoys before dumping them as pdb
        save_log (bool): save scores to logfile at <output_dir/scores.txt>
    """
    cfg = abinitio_cfg
    cfg.update_config(n_cores=n_cores)

    pool = multiprocessing.Pool()
    pool_outputs = pool.map(_create_decoys, range(cfg.n_cores))

    # reset log for multiprocess run
    if save_log:
        log_file = f"{output_dir}/scores.txt"
        if os.path.exists(log_file):
            os.remove(log_file)

    if cfg.n_decoys >= cfg.n_cores:
        print(f">>> Decoy creation will be distributed to {cfg.n_cores} cores.")
    else:
        print(f"""CFG file issue:
<n_decoys parameter>: {cfg.n_decoys}
<n_cores parameter>: {cfg.n_cores}""")
        print(f">>> Decoy creation will be distributed to use only {cfg.n_decoys} cores.")
        cfg.n_cores = cfg.n_decoys

    for worker in range(cfg.n_cores):
        cfg = cfg.deepcopy()
        cfg.decoy_ndx_shift = worker * int(cfg.n_decoys/cfg.n_cores)
        args = [cfg, output_dir, stream2pymol, fastrelax, save_log]
        pool.map(_create_decoys, args)

    pool.close()
    pool.join()
    return


def _create_decoys(abinitio_cfg, output_dir="./output",
                   stream2pymol=True, fastrelax=True, save_log=True):
    """
    Create decoys within pyrosetta framework.

    Args:
        abinitio_cfg (CONFIG class): output of abinitio.setup_abinitio_cfg()
        output_dir (str): output directory for decoys
        stream2pymol (bool): stream decoys to pymol
        fastrelax (bool): apply fastrelax protocol on decoys before dumping them as pdb
        save_log (bool): save scores to logfile at <output_dir/scores.txt>

    Returns:
        SCORES_low (list): centroid scores ~ score 3
        SCORES_high (list): fa scores ~ ref2015
    """
    cfg = abinitio_cfg

    ### create output directory
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    if os.path.exists(output_dir) and cfg.n_cores == 1:
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
    #jd = PyJobDistributor(cfg.job_name, cfg.n_decoys, scorefxn_high)

    if stream2pymol:
        pmm = pyrosetta.PyMOLMover()
        pmm.keep_history(True)

    ### job distributor stuff
    worker_jobs = int(cfg.n_decoys/cfg.n_cores)
    #SCORES_low = [0]*(worker_jobs)  # scores array
    #SCORES_high = [0]*(worker_jobs)  # scores array
    DECOYS = []
    SCORES_low = []
    SCORES_high = []
    for i in range(1, worker_jobs+1):
        _misc.cprint(f">>> Working on decoy: {output_dir}/{cfg.job_name}_{cfg.decoy_ndx_shift+i}.pdb", "blue")
        DECOYS.append(f"{cfg.job_name}_{cfg.decoy_ndx_shift+i}")
        pose.assign(pose_0)
        pose.pdb_info().name(f"{cfg.job_name}_{cfg.decoy_ndx_shift+i}")
        mc.reset(pose)
        for j in range(cfg.folding_repeats):
            folding.apply(pose)
            mc.recover_low(pose)
        #SCORES_low[i] = scorefxn_low(pose)
        SCORES_low.append(scorefxn_low(pose))
        to_fullatom.apply(pose)

        if fastrelax:
            relax = protocols.relax.FastRelax()
            relax.set_scorefxn(scorefxn_high)
            relax.apply(pose)

        #SCORES_high[i] = scorefxn_high(pose)
        SCORES_high.append(scorefxn_high(pose))
        pose.dump_pdb(f"{output_dir}/{cfg.job_name}_{cfg.decoy_ndx_shift+i}.pdb")

        if stream2pymol:
            pmm.apply(pose)

    if save_log:
        logfile = f"{output_dir}/scores.txt"
        if not os.path.exists(logfile):
            with open(logfile, "w") as log:
                log.write(f"{'decoy'}\t{'score3'}\t{'ref2015'}\n")  # write header
        with open(logfile, "a") as log:
            #DECOYS = [f"{cfg.job_name}_{cfg.decoy_ndx_shift+i}" for i in range(worker_jobs)]
            table_str = _misc.print_table([DECOYS, SCORES_low, SCORES_high])
            log.write(table_str)

    return SCORES_low, SCORES_high


def _HELP_decoypath_to_decoyid(list_):
    """
    HELP FUNCTION: convert list with decoy strings to list with decoy ids

    Example:
        decoy_list = ['./1LMB_decoys/set1_2000/1LMB_1.pdb',
                      './1LMB_decoys/set1_2000/1LMB_2.pdb',
                      './1LMB_decoys/set1_2000/1LMB_3.pdb']
        _HELP_decoystr_to_decoyid(decoy_list)
        >> [1, 2, 3]
    """
    substrings = [_misc.get_substrings(i) for i in list_]

    # get decoy index range
    diff_element = list(set(substrings[0])-set(substrings[1]))
    if len(diff_element) != 1:
        raise ValueError('substrings differ in more than 1 element (i.e. differ at more than 1 position)')
    diff_position = substrings[0].index(diff_element[0])
    ids = [int(_misc.get_substrings(i)[diff_position]) for i in list_]
    return ids


def get_decoy_precission(ref, decoy_dir, pattern="*.pdb",
                         ndx_range=(None, None), sel='backbone'):
    """
    Args:
        ref (MDA universe): reference structure
        decoy_dir (str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple/list): limit decoy index range
            ndx_range[0]: ndx_min
            ndx_range[1]: ndx_max
        sel (str): selection string

    Returns:
        DECOY_LIST (list): list with decoy paths
        DECOY_ID   (list): list with decoy ids
        SCORE (list): list with corresponding scores (ref 2015)
        RMSD  (list): list with corresponding RMSD values
    """
    scorefxn = pyrosetta.get_fa_scorefxn()
    DECOY_LIST = get_decoy_list(decoy_dir, pattern, ndx_range)
    DECOY_ID = _HELP_decoypath_to_decoyid(DECOY_LIST)
    SCORE = []  # list with corresponding scores
    RMSD = []   # list with corresponding RMSD

    for decoy_path in tqdm(DECOY_LIST):
        if not os.path.exists(decoy_path):
            print(f"The decoy path '{decoy_path}' does not exist!")
            continue

        # get SCORE
        pose = pyrosetta.pose_from_pdb(decoy_path)
        SCORE.append(scorefxn(pose))

        # get RMSD
        decoy = mda.Universe(decoy_path)
        sel1, sel2 = _ana.get_matching_selection(decoy, ref, sel=sel, norm=True, verbose=False)
        ftr = _ana.get_RMSD(decoy, ref, sel1=sel1, sel2=sel2)
        RMSD.append(ftr[2])

    RMSD = _misc.flatten_array(RMSD)
    return DECOY_LIST, DECOY_ID, SCORE, RMSD


def rank_decoy_precission(data, rank_by="SCORE", verbose=True):
    """
    Args:
        data (list): output of abinitio.get_decoy_precission()
            data[0]: DECOY_LIST
            data[1]: DECOY_ID
            data[2]: SCORE
            data[3]: RMSD
        rank_by (str): "SCORE", "RMSD"
        verbose (bool)

    Returns:
        DECOY_LIST_ranked (array)
        DECOY_ID_ranked (array)
        SCORE_ranked (array)
        RMSD_ranked (array)
    """
    DECOY_LIST, DECOY_ID, SCORE, RMSD = data
    if rank_by.upper() == "SCORE":
        SCORE_ranked, SCORE_ranked_ndx = _misc.get_ranked_array(SCORE, reverse=True, verbose=verbose)
        DECOY_LIST_ranked = np.array([DECOY_LIST[ndx] for ndx in SCORE_ranked_ndx])
        DECOY_ID_ranked = np.array([DECOY_ID[ndx] for ndx in SCORE_ranked_ndx])
        RMSD_ranked = np.array([RMSD[ndx] for ndx in SCORE_ranked_ndx])
    if rank_by.upper() == "RMSD":
        RMSD_ranked, RMSD_ranked_ndx = _misc.get_ranked_array(RMSD, reverse=True, verbose=verbose)
        DECOY_LIST_ranked = np.array([DECOY_LIST[ndx] for ndx in RMSD_ranked_ndx])
        DECOY_ID_ranked = np.array([DECOY_ID[ndx] for ndx in RMSD_ranked_ndx])
        SCORE_ranked = np.array([SCORE[ndx] for ndx in RMSD_ranked_ndx])

    return DECOY_LIST_ranked, DECOY_ID_ranked, SCORE_ranked, RMSD_ranked


def precission_scatterplot(SCORE, RMSD, **kwargs):
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
    plt.tight_layout()
    return fig, ax
