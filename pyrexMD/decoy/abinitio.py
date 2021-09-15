# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: abinitio.py
# @Last modified by:   arthur
# @Last modified time: 15.09.2021

"""
.. hint:: This module contains functions for decoy creation using `PyRosetta`.
"""


from tqdm.notebook import tqdm
import pyrexMD.misc as _misc
import pyrexMD.topology as _top
import pyrexMD.analysis.analyze as _ana
from pyrexMD.analysis.analyze import get_Distance_Matrices
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import multiprocessing
import MDAnalysis as mda
import pyrosetta
from pyrosetta.rosetta import protocols
pyrosetta.init("-mute basic.io.database core.import_pose core.scoring core.chemical.GlobalResidueTypeSet core.pack.task core.pack.pack_rotamers core.pack.dunbrack.RotamerLibrary protocols.relax.FastRelax core.pack.interaction_graph.interaction_graph_factory")  # mute messages


def frame2score(mobile, frames, save_as, sel="protein", warn=True):
    """
    Get rosetta scores (ref2015) for frames of mobile.trajectory

    Args:
        mobile
        frames (list): frame indices of mobile.trajectory
        save_as (str): logfile containing table of [FRAMES, SCORES]
        sel (str): selection string

    Returns:
        FRAMES (list)
            list with frames of mobile.trajectory
        SCORES (list)
            list with rosetta scores (ref2015)
    """
    if warn:
        _misc.cprint("""Warning: selection should be 'protein' in order to work with full atom rosetta scores. But you can still reduce the selection string to ignore parts of the protein, e.g 'protein and not resids 1-8' etc.""", "red")
    if isinstance(mobile, str):
        mobile = mda.Universe(mobile)
    _top.norm_universe(mobile)

    scorefxn_high = pyrosetta.create_score_function('ref2015')
    to_fullatom = pyrosetta.SwitchResidueTypeSetMover('fa_standard')

    FRAMES = []
    SCORES = []

    for frame in tqdm(frames):
        mobile.trajectory[frame]
        structure = mobile.select_atoms(sel)
        structure.write("temp.pdb")

        pose = pyrosetta.pose_from_pdb("temp.pdb")
        to_fullatom.apply(pose)

        # high score
        relax = protocols.relax.FastRelax()
        relax.set_scorefxn(scorefxn_high)
        relax.apply(pose)
        SCORES.append(scorefxn_high(pose))

        # frame
        FRAMES.append(frame)

    _misc.rm("temp.pdb", verbose=False)

    with open(save_as, "w") as fout:
        fout.write(_misc.print_table([FRAMES, SCORES], verbose=False))
    return FRAMES, SCORES


def get_torsion_angles(obj, mol_type="auto", prec=2, spacing=12, verbose=True, verbose_stop=None):
    """
    get torsion angles of object.

    Args:
        obj (pose, str): pyrosetta pose object or pdb path
        mol_type (str):
          | molecule type
          | "auto": detect "protein" or "rna" by checking if pose.phi(1) exists.
          | "protein": protein molecule with torsion angles(phi, psi)
          | "rna": rna molecule with torsion angles(alpha, beta, gamma, delta)
        prec (int): rounding precision
        verbose (bool): print table with torsion angles

    Returns:
        TORSION_ANGLES (tuple)
          | tuple containing lists with torsion angles of each residue
          | if mol_type == "protein":
          | TORSION_ANGLES = (phi, psi)
          | if mol_type == "rna":
          | TORSION_ANGLES = (alpha, beta, gamma, delta)
    """
    if isinstance(obj, str):
        pose = pyrosetta.pose_from_pdb(obj)
    else:
        pose = obj

    if mol_type.lower() == "auto":
        try:
            pose.phi(1)
            mol_type = "protein"
        except RuntimeError:
            mol_type = "rna"

    if mol_type.lower() == "protein":
        PHI, PSI, = [], []

        for i in range(1, 1+pose.total_residue()):
            PHI.append(round(pose.phi(i), prec))
            PSI.append(round(pose.psi(i), prec))

        TORSION_ANGLES = (PHI, PSI)
        if verbose:
            _misc.cprint("resid \tphi \tpsi".expandtabs(spacing), "blue")
            RESID = list(range(1, 1+pose.total_residue()))
            _misc.print_table([RESID, PHI, PSI],
                              prec=prec, spacing=spacing, verbose_stop=verbose_stop)

    if mol_type.lower() == "rna":
        ALPHA, BETA, GAMMA, DELTA = [], [], [], []

        for i in range(1, 1+pose.total_residue()):
            ALPHA.append(round(pose.alpha(i), prec))
            BETA.append(round(pose.beta(i), prec))
            GAMMA.append(round(pose.gamma(i), prec))
            DELTA.append(round(pose.delta(i), prec))

        TORSION_ANGLES = (ALPHA, BETA, GAMMA, DELTA)
        if verbose:
            _misc.cprint("resid  \talpha \tbeta \tgamma \tdelta".expandtabs(spacing), "blue")
            RESID = list(range(1, 1+pose.total_residue()))
            _misc.print_table([RESID, ALPHA, BETA, GAMMA, DELTA],
                              prec=prec, spacing=spacing, verbose_stop=verbose_stop)

    return TORSION_ANGLES


def apply_relative_torsion_angle(pose_ref, pose, resid=1, angle="alpha", shift=0, verbose=True):
    """
    Apply relative torsion angle(use pose_ref angle and apply relative shift).

    Args:
        pose_ref (pose)
        pose (pose)
        resid (int)
        angle (str)
          | protein: "phi", "psi"
          | rna: "alpha", "beta", "gamma", "delta"
        shift (float): relative shift value
        verbose (bool): print applied changes to torsion angle.
    """
    with _misc.HiddenPrints(verbose=verbose):
        if angle.lower() == "phi":
            set_angle = pose.set_phi
            vref = pose_ref.phi(resid)
            v1 = pose.phi(resid)
        elif angle.lower() == "psi":
            set_angle = pose.set_psi
            vref = pose_ref.psi(resid)
            v1 = pose.psi(resid)
        elif angle.lower() == "alpha":
            set_angle = pose.set_alpha
            vref = pose_ref.alpha(resid)
            v1 = pose.alpha(resid)
        elif angle.lower() == "beta":
            set_angle = pose.set_beta
            vref = pose_ref.beta(resid)
            v1 = pose.beta(resid)
        elif angle.lower() == "gamma":
            set_angle = pose.set_gamma
            vref = pose_ref.gamma(resid)
            v1 = pose.gamma(resid)
        elif angle.lower() == "delta":
            set_angle = pose.set_delta
            vref = pose_ref.delta(resid)
            v1 = pose.delta(resid)

        v2 = vref + shift
        set_angle(resid, v2)
        print(f"resid {resid} {angle:>5}: {v1:>8.2f}  --> {v2:>8.2f}  (ref:{vref:>8.2f})")
    return
################################################################################
################################################################################


def setup_abinitio_cfg(pdbid, fasta_seq, frag3mer, frag9mer, **kwargs):
    """
    Setup config file for abinitio functions.

    Args:
        pdbid (str): pdb id / pdb name
        fasta_seq (str): fasta sequence
        frag3mer (str): 3mer file path(get via robetta server)
        frag9mer (str): 9mer file path(get via robetta server)

    Keyword Args:
        fasta_seq_len (int): fasta sequence length
        frag3inserts (int): number of frag3 inserts
        frag9inserts (int): number of frag9 inserts
        folding_cycles (int): folding move cycles
        folding_repeats (int): folding move repeats
        job_name (str)
        n_decoys (int): total number of decoys
        n_cores (int): -np option for multiprocessing
        decoy_ndx_shift (int):
          | shift decoy index(output filename) by this value
          | required for multiprocessing to fix names of decoys
        kT (float): kT parameter during Monte-Carlo simulation

    Returns:
        abinitio_cfg (CONFIG class)
            configs used as input for abinitio.create_decoys()
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
    abinitio_cfg = _misc.CONFIG(default)
    abinitio_cfg.update_config(fasta_seq_len=len(abinitio_cfg.fasta_seq))
    abinitio_cfg.update_config(**kwargs)
    return abinitio_cfg


def create_decoys(abinitio_cfg, output_dir="./output",
                  stream2pymol=True, fastrelax=True, save_log=True):
    """
    Create decoys within pyrosetta framework.

    .. Warning: : this function is currently broken (but abinitio._create_decoys()
                 is working with 1 core)

    Args:
        abinitio_cfg (CONFIG class): output of abinitio.setup_abinitio_cfg()
        output_dir (str): output directory for decoys
        stream2pymol (bool): stream decoys to pymol
        fastrelax (bool): apply fastrelax protocol on decoys before dumping them as pdb
        save_log (bool): save scores to logfile at < output_dir/scores.txt >
    """
    cfg = abinitio_cfg

    pool = multiprocessing.Pool()

    # reset log for multiprocess run
    if save_log:
        log_file = f"{output_dir}/scores.txt"
        if os.path.exists(log_file):
            os.remove(log_file)

    if cfg.n_decoys >= cfg.n_cores:
        print(f">>> Decoy creation will be distributed to {cfg.n_cores} cores.")
    else:
        print(f"""CFG file issue:
<n_decoys parameter >: {cfg.n_decoys}
<n_cores parameter >: {cfg.n_cores}""")
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
                   stream2pymol=True, fastrelax=True, save_log=True, **kwargs):
    """
    Create decoys within pyrosetta framework.

    Args:
        abinitio_cfg (CONFIG class): output of abinitio.setup_abinitio_cfg()
        output_dir (str): output directory for decoys
        stream2pymol (bool): stream decoys to pymol
        fastrelax (bool): apply fastrelax protocol on decoys before dumping them as pdb
        save_log (bool): save scores to logfile at < output_dir/scores.txt >

    Keyword Args:
        cprint_color (None, str): colored print color

    Returns:
        SCORES_low (list)
            centroid scores ~ score 3
        SCORES_high (list)
            fa scores ~ ref2015
    """
    default = {"cprint_color": "blue"}
    default_cfg = _misc.CONFIG(default, **kwargs)
    cfg = _misc.CONFIG(abinitio_cfg, **default_cfg)

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
        _misc.cprint(f">>> Working on decoy: {output_dir}/{cfg.job_name}_{cfg.decoy_ndx_shift+i}.pdb", cfg.cprint_color)
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
    convert list with decoy strings to list with decoy ids

    Example:
      | >> decoy_list = ['./1LMB_decoys/set1_2000/1LMB_1.pdb',
      |                 './1LMB_decoys/set1_2000/1LMB_2.pdb',
      |                 './1LMB_decoys/set1_2000/1LMB_3.pdb']
      | >> _HELP_decoystr_to_decoyid(decoy_list)
      | [1, 2, 3]
    """
    substrings = [_misc.get_substrings(i) for i in list_]

    # get decoy index range
    diff_element = list(set(substrings[0])-set(substrings[1]))
    if len(diff_element) != 1:
        raise ValueError('substrings differ in more than 1 element (i.e. differ at more than 1 position)')
    diff_position = substrings[0].index(diff_element[0])
    ids = [int(_misc.get_substrings(i)[diff_position]) for i in list_]
    return ids


def get_decoy_list(decoy_dir, pattern="*.pdb", ndx_range=(None, None)):
    """
    | Alias function of get_structure_list().
    | get decoy list (sorted by a numeric part at any position of the filename,
    | e.g. 1LMB_1.pdb, 1LMB_2.pdb, ...)

    Args:
        decoy_dir (str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple, list): limit decoy index range to [ndx_min, ndx_max]

    Returns:
        DECOY_LIST (list)
            list with decoy filenames
    """
    if not isinstance(decoy_dir, str):
        raise TypeError("type(decoy_dir) must be string.")
    if decoy_dir[-1] == "/":
        decoy_dir = decoy_dir[:-1]

    # preset stuff
    decoy_list = glob.glob(f"{decoy_dir}/{pattern}")
    decoy_names = [_misc.get_filename(i) for i in decoy_list]
    decoy_names_substrings = [_misc.get_substrings(i) for i in decoy_names]

    # get decoy index range
    diff_element = list(set(decoy_names_substrings[0])-set(decoy_names_substrings[1]))
    if len(diff_element) != 1:
        raise ValueError('decoy_names_substrings differ in more than 1 element (i.e. differ at more than 1 position)')
    diff_position = decoy_names_substrings[0].index(diff_element[0])
    min_ndx = 999999999
    max_ndx = -999999999
    for i in decoy_names_substrings:
        if i[diff_position].isdigit() and int(i[diff_position]) < min_ndx:
            min_ndx = int(i[diff_position])
        if i[diff_position].isdigit() and int(i[diff_position]) > max_ndx:
            max_ndx = int(i[diff_position])

    if ndx_range[0] is not None:
        min_ndx = ndx_range[0]
    if ndx_range[1] is not None:
        max_ndx = ndx_range[1]

    # get decoy filename prefix and suffix
    template = decoy_names[0]
    template_diff_element = _misc.get_substrings(template)[diff_position]
    decoy_filename_prefix = template[:template.index(template_diff_element)]
    decoy_filename_suffix = template[template.index(template_diff_element)+len(template_diff_element):]

    # create sorted decoy list
    DECOY_LIST = [f"{decoy_dir}/{decoy_filename_prefix}{i}{decoy_filename_suffix}" for i in range(min_ndx, max_ndx+1)
                  if f"{decoy_dir}/{decoy_filename_prefix}{i}{decoy_filename_suffix}" in decoy_list]
    return DECOY_LIST


### Alias function of get_decoy_list()
def get_structure_list(structure_dir, pattern="*.pdb", ndx_range=(None, None)):
    """
    | Alias function of get_decoy_list().
    | get structure list(sorted by a numeric part at any position of the filename,
    | e.g. 1LMB_1.pdb, 1LMB_2.pdb, ...)

    Args:
        structure_dir (str): structure directory
        pattern (str): pattern of structure filenames
        ndx_range (tuple, list): limit structure index range to[ndx_min, ndx_max]

    Returns:
        STRUCTURE_LIST (list)
            list with structure filenames
    """
    return(get_decoy_list(decoy_dir=structure_dir, pattern=pattern, ndx_range=ndx_range))


def get_decoy_scores(decoy_list=None, decoy_dir=None, pattern="*.pdb",
                     ndx_range=(None, None), n_rank=None, reverse=True,
                     verbose=True, **kwargs):
    """
    Get decoy scores of < decoy_list > or < decoy_dir > .

      - If < n_rank > is None: return unranked decoys, ids and scores.
      - If < n_rank > is int: return ranked decoys, ids and scores for best < n_rank > decoys.

    Args:
        decoy_list (None, list): list with decoy paths
        decoy_dir (None, str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple, list): limit decoy index range to[ndx_min, ndx_max]
        n_rank (None, int):
          | None: return unranked decoys, ids and scores.
          | int: return ranked decoys, ids and scores for best < n_rank > decoys.
        reverse (bool):
          | True: ascending ranking order(low to high)
          | False: decending ranking order(high to low)
        verbose (bool)

    Keyword Args:
        cprint_color (None, str): colored print color

    Returns:
        DECOY_LIST (list)
            list with decoy paths
        DECOY_ID (list)
            list with decoy ids
        SCORE (list)
            list with corresponding scores(ref 2015)
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    scorefxn = pyrosetta.get_fa_scorefxn()
    if decoy_list is None and decoy_dir is None:
        raise TypeError("Specify either decoy_list or decoy_dir.")
    elif isinstance(decoy_list, list):
        DECOY_LIST = decoy_list
    elif isinstance(decoy_dir, str):
        DECOY_LIST = get_decoy_list(decoy_dir, pattern, ndx_range)
    else:
        raise TypeError("Specify either decoy_list or decoy_dir.")
    DECOY_ID = _HELP_decoypath_to_decoyid(DECOY_LIST)
    SCORE = []  # list with corresponding scores

    if verbose:
        _misc.cprint("Getting scores...", cfg.cprint_color)
    for decoy_path in tqdm(DECOY_LIST, disable=not verbose):
        if not os.path.exists(decoy_path):
            print(f"The decoy path '{decoy_path}' does not exist!")
            continue

        # get SCORE
        pose = pyrosetta.pose_from_pdb(decoy_path)
        SCORE.append(scorefxn(pose))

    if n_rank is not None:
        DECOY_LIST = [DECOY_LIST[i] for i in _misc.get_ranked_array(SCORE, reverse=reverse, verbose=False)[1][:n_rank]]
        DECOY_ID = [DECOY_ID[i] for i in _misc.get_ranked_array(SCORE, reverse=reverse, verbose=False)[1][:n_rank]]
        SCORE = [SCORE[i] for i in _misc.get_ranked_array(SCORE, reverse=reverse, verbose=False)[1][:n_rank]]
    return DECOY_LIST, DECOY_ID, SCORE


def get_decoy_RMSD(ref, decoy_list=None, decoy_dir=None, pattern="*.pdb",
                   ndx_range=(None, None), n_rank=None, reverse=True,
                   sel='backbone', verbose=True, **kwargs):
    """
    Get decoy RMSD of < decoy_list > or < decoy_dir > .

      - If < n_rank > is None: return unranked decoys, ids and RMSD.
      - If < n_rank > is int: return ranked decoys, ids and RMSD for best < n_rank > decoys.

    Args:
        ref (universe): reference structure
        decoy_list (None, list): list with decoy paths
        decoy_dir (None, str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple, list): limit decoy index range to[ndx_min, ndx_max]
        n_rank (None, int):
          | None: return unranked decoys, ids and scores.
          | int: return ranked decoys, ids and scores for best < n_rank > decoys.
        reverse (bool):
          | True: ascending ranking order(low to high)
          | False: decending ranking order(high to low)
        verbose (bool)

    Keyword Args:
        cprint_color (None, str): colored print color

    Returns:
        DECOY_LIST (list)
            list with decoy names
        DECOY_ID (list)
            list with decoy ids
        RMSD (list)
            list with decoy RMSDs
    """
    default = {"cprint_color": "blue"}
    cfg = _misc.CONFIG(default, **kwargs)
    ############################################################################
    if decoy_list is None and decoy_dir is None:
        raise TypeError("Specify either decoy_list or decoy_dir.")
    elif isinstance(decoy_list, list):
        DECOY_LIST = decoy_list
    elif isinstance(decoy_dir, str):
        DECOY_LIST = get_decoy_list(decoy_dir, pattern, ndx_range)
    else:
        raise TypeError("Specify either decoy_list or decoy_dir.")
    DECOY_ID = _HELP_decoypath_to_decoyid(DECOY_LIST)
    RMSD = []  # list with corresponding RMSD values

    if verbose:
        _misc.cprint("Getting RMSDs...", cfg.cprint_color)
    for decoy_path in tqdm(DECOY_LIST, disable=not verbose):
        if not os.path.exists(decoy_path):
            print(f"The decoy path '{decoy_path}' does not exist!")
            continue

        # get RMSD
        decoy = mda.Universe(decoy_path, dt=0)  # dt=0 fixes warnings.warn("Reader has no dt information, set to 1.0 ps") of MDAnalysis/coordinates/base.py
        sel1, sel2 = _ana.get_matching_selection(decoy, ref, sel=sel, norm=True, verbose=False)
        ftr = _ana.get_RMSD(decoy, ref, sel1=sel1, sel2=sel2)
        RMSD.append(ftr[2])

    RMSD = _misc.flatten_array(RMSD)

    if n_rank is not None:
        DECOY_LIST = [DECOY_LIST[i] for i in _misc.get_ranked_array(RMSD, reverse=reverse, verbose=False)[1][:n_rank]]
        DECOY_ID = [DECOY_ID[i] for i in _misc.get_ranked_array(RMSD, reverse=reverse, verbose=False)[1][:n_rank]]
        RMSD = [RMSD[i] for i in _misc.get_ranked_array(RMSD, reverse=reverse, verbose=False)[1][:n_rank]]
    return DECOY_LIST, DECOY_ID, RMSD


def get_decoy_precision(ref, decoy_list=None, decoy_dir=None, pattern="*.pdb",
                        ndx_range=(None, None), sel='backbone', verbose=True):
    """
    get decoy precision

    Args:
        ref (universe): reference structure
        decoy_dir (str): decoy directory
        pattern (str): pattern of decoy filenames
        ndx_range (tuple, list): limit decoy index range to[ndx_min, ndx_max]
        sel (str): selection string(for RMSD calculation)
        verbose (bool)

    Returns:
        DECOY_LIST (list)
            list with decoy paths
        DECOY_ID (list)
            list with decoy ids
        SCORE (list)
            list with corresponding scores(ref 2015)
        RMSD (list)
            list with corresponding RMSD values
    """
    DECOY_LIST, DECOY_ID, SCORE = get_decoy_scores(decoy_dir=decoy_dir, pattern=pattern, ndx_range=ndx_range, verbose=verbose)
    DECOY_LIST, DECOY_ID, RMSD = get_decoy_RMSD(ref=ref, decoy_list=decoy_list, decoy_dir=decoy_dir, pattern=pattern, ndx_range=ndx_range, verbose=verbose)
    return DECOY_LIST, DECOY_ID, SCORE, RMSD


def rank_decoy_precision(data, rank_by="SCORE", verbose=True):
    """
    rank decoy precision

    Args:
        data (list):
          | output of abinitio.get_decoy_precision()
          | data[0]: DECOY_LIST
          | data[1]: DECOY_ID
          | data[2]: SCORE
          | data[3]: RMSD
        rank_by (str): "SCORE", "RMSD"
        verbose (bool)

    Returns:
        DECOY_LIST_ranked (array)
            ranked list with decoy paths
        DECOY_ID_ranked (array)
            ranked list with decoy ids
        SCORE_ranked (array)
            ranked list with corresponding scores (ref 2015)
        RMSD_ranked (array)
            ranked list with corresponding RMSD values
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


def precision_scatterplot(SCORE, RMSD, **kwargs):
    """
    create a precision scatter plot.

    Args:
        SCORE (list, array)
        RMSD (list, array)

    .. Hint: : Args and Keyword Args of misc.figure() are valid Keyword Args.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """

    fig, ax = _misc.figure(**kwargs)
    plt.plot(RMSD, SCORE, ".")
    plt.xlabel(r"RMSD ($\AA$)", fontweight="bold")
    plt.ylabel("SCORE (REU)", fontweight="bold")
    plt.tight_layout()
    return fig, ax
