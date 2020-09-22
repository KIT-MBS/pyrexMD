from __future__ import division, print_function
from builtins import next
import myPKG.misc as _misc
import myPKG.analysis as _ana
import numpy as np


def parsePDB_RES_ATOM_NAME(fin, skiprows="auto", ignh=True):
    """
    Reads PDB file and returns the columns for residue, atom-number and atom-name as lists.
    Removes column entries which consist only of whitespace or newlines.

    Args:
        fin (str): PDB file
        skiprows (int): ignore header rows of fin
            -1 or "auto": auto detect
        ignh (bool): ignore hydrogen

    Returns:
        RES (list): residue column of PDB
        ATOM (list): atom column of PDB
        NAME (list): name column of PDB
    """
    # help function to ignh
    def HELP_func_ignh(line):
        """
        help function: ignore hydrogen

        If last char of line is numeric: return False  -> atom is hydrogen
        If last char of line is  letter: return True   -> atom is heavy atom (C,N,O,S,...)
        """
        # ignore new line entries
        if line[-1] != "\n":
            test_char = line[-1]
        else:
            test_char = line[-2]

        if test_char.isnumeric():
            # last char is numeric -> atom is hydrogen
            return False
        else:
            # last char is letter -> atom is heavy
            return True

    with open(fin, "r") as f:

        RES = []
        ATOM = []
        NAME = []

        # skip header rows
        if skiprows == -1 or skiprows == "auto":
            skiprows = _misc.autodetect_header(fin)
        for i in range(skiprows):
            next(f)
        for line in f:
            chars = list(line)

            # concatenate single chars to string
            concat_RES = ''.join(chars[22:26])
            concat_ATOM = ''.join(chars[6:11])
            concat_NAME = ''.join(chars[12:16])
            # remove all whitespace (empty colums ~ PDB format)
            concat_RES = concat_RES.replace(" ", "")
            concat_ATOM = concat_ATOM.replace(" ", "")
            concat_NAME = concat_NAME.replace(" ", "")
            # append to list
            if ignh == False:
                # append all
                RES.append(concat_RES)
                ATOM.append(concat_ATOM)
                NAME.append(concat_NAME)
            elif HELP_func_ignh(chars):
                # append only heavy atoms
                RES.append(concat_RES)
                ATOM.append(concat_ATOM)
                NAME.append(concat_NAME)

        # remove empty entries (might occur at beginning/ending of structures)
        RES = [int(item) for item in RES if item.isalnum()]
        ATOM = [int(item) for item in ATOM if item.isalnum()]
        NAME = [item for item in NAME if item.isalnum()]

    return RES, ATOM, NAME


def DCAREX_res2atom(ref_pdb, DCA_fin, n_DCA, usecols, n_bonds=1, ref_skiprows="auto", DCA_skiprows="auto",
                    filter_DCA=True, save_log=True, logFileName="PDBID_DCA_used.txt"):
    """
    Return lists of matching RES pairs and ATOM pairs for "DCA REX Workflow"

    Algorithm:
        - Read files line by line, skip header if skiprows is set correctly
            -- ref_pdb: get RES, ATOM, NAME
            -- DCA_fin: get DCA_pairs
        - for IJ in DCA_pairs:
            -- find matching CA atoms (CA_ij)
            -- append IJ to RES_pair, CA_ij to ATOM_pair
        - return RES_pair and ATOM_pair

    Args:
        ref_pdb (str): reference PDB (path)
        ref_skiprows (int): ignore header rows of ref
            -1 or "auto": auto detect
        DCA_fin (str): DCA file (path)
        n_DCA (int): number of DCA contacts which should be used
        usecols (tuple/list): columns containing the RES pairs in DCA_fin
        n_bonds (int/str): number of restraints per DCA contact
            "all", 0 or -1: bonds between all heavy atoms of RES pair ij
                    -> all permutations of heavy atoms per DCA pair
            1: bonds between  CA atoms of RES pair ij
                    -> 1 bond per DCA pair
            2: bonds between CA, CB atoms of RES pair ij
                    -> 2 bonds per DCA pair
            3: bonds between CA, CB, CG atoms of RES pair ij
                    -> 3 bonds per DCA pair
        DCA_skiprows (int): ignore header rows of DCA_fin
            -1 or "auto": auto detect
        filter_DCA (bool):
            True: ignore DCA pairs with |i-j| < 3
            False: use all DCA pairs w/o applying filter
        save_log (bool)
        logFileName (str):
            if "PDBID" in logFileName: detect PDBID based on ref_PDB path
    Returns:
        RES_PAIR (list)
        ATOM_PAIR (list)
    """
    print('ATTENTION: Make sure that ref_pdb is the reference PDB taken after applying a forcefield, since added hyrogen atoms will shift the atom numbers.\n')

    # read files
    RES, ATOM, NAME = parsePDB_RES_ATOM_NAME(ref_pdb, ref_skiprows)
    DCA_PAIR, _ = _misc.read_DCA_file(DCA_fin, n_DCA, usecols=usecols, skiprows=DCA_skiprows, filter_DCA=filter_DCA)

    RES_PAIR = []
    ATOM_PAIR = []

    # warning
    if n_bonds not in ["all", -1, 0, 1, 2, 3]:
        print("""DCAREX_res2atom warning: invalid n_bonds parameter. Valid values are ["all",-1,0,1,2,3]. Set n_bonds to 1.""")
        n_bonds = 1

    # case1: all atoms
    if n_bonds == "all" or n_bonds == 0 or n_bonds == -1:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            ATOM1 = ATOM[_temp_min:_temp_max]

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            ATOM2 = ATOM[_temp_min:_temp_max]
            RES_PAIR.append((IJ[0], IJ[1]))

            TEMP = []
            for item1 in ATOM1:
                for item2 in ATOM2:
                    TEMP.append((item1, item2))
                ATOM_PAIR.append(TEMP)

    # case2: CA atoms
    elif n_bonds == 1:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append((ATOM[CA1_index], ATOM[CA2_index]))

    # case3: CA and CB atoms
    elif n_bonds == 2:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append([(ATOM[CA1_index], ATOM[CA2_index]),
                              (ATOM[CB1_index], ATOM[CB2_index])])
    # case4: CA, CB and CG atoms
    elif n_bonds == 3:
        for IJ in DCA_PAIR:
            _temp_min = RES.index(IJ[0])
            _temp_max = RES.index(IJ[0]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA1_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB1_index = _temp_min + _temp_index
            if "CG" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG')
                CG1_index = _temp_min + _temp_index
            elif "CG1" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG1')
                CG1_index = _temp_min + _temp_index
            elif "CG2" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG2')
                CG1_index = _temp_min + _temp_index
            elif "C"in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('C')
                CG1_index = _temp_min + _temp_index

            _temp_min = RES.index(IJ[1])
            _temp_max = RES.index(IJ[1]+1)
            _temp_index = NAME[_temp_min:_temp_max].index('CA')
            CA2_index = _temp_min + _temp_index
            _temp_index = NAME[_temp_min:_temp_max].index('CB')
            CB2_index = _temp_min + _temp_index
            if "CG" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG')
                CG2_index = _temp_min + _temp_index
            elif "CG1" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG1')
                CG2_index = _temp_min + _temp_index
            elif "CG2" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('CG2')
                CG2_index = _temp_min + _temp_index
            elif "C" in NAME[_temp_min:_temp_max]:
                _temp_index = NAME[_temp_min:_temp_max].index('C')
                CG2_index = _temp_min + _temp_index

            RES_PAIR.append((RES[CA1_index], RES[CA2_index]))
            ATOM_PAIR.append([(ATOM[CA1_index], ATOM[CA2_index]),
                              (ATOM[CB1_index], ATOM[CB2_index]),
                              (ATOM[CG1_index], ATOM[CG2_index])])
    if save_log:
        new_dir = _misc.mkdir('logs')
        if "PDBID" in logFileName:
            pdbid = _misc.get_PDBid(ref_pdb)
            logFileName = pdbid + "_DCA_used.txt"
            logfile = new_dir + "/" + logFileName
        print("Saved log as:", logfile)
        with open(logfile, "w") as fout:
            if n_bonds in [1, 2, 3]:
                fout.write("#DCA contacts top{} ({} bonds per contact)\n".format(n_DCA, n_bonds))
            else:
                fout.write("#DCA contacts top{} (all heavy atom bonds per conact)\n".format(n_DCA))
            fout.write("{}\t{}\t{}\t{}\n".format("#RESi", "RESj", "ATOMi", "ATOMj"))
            for i in range(len(RES_PAIR)):
                if type(ATOM_PAIR[i]) == tuple:
                    fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], ATOM_PAIR[i][0], ATOM_PAIR[i][1]))
                else:
                    for item in ATOM_PAIR[i]:
                        fout.write("{}\t{}\t{}\t{}\n".format(RES_PAIR[i][0], RES_PAIR[i][1], item[0], item[1]))
    return RES_PAIR, ATOM_PAIR


def DCAREX_modify_scoreFile(score_fin, shift_res, outputFileName="PDBID_mod.score", res_cols=(2, 3), score_col=(5)):
    """
    Modify score file (MSA scores) by shifting residues.

    Args:
        score_fin (str): path to score file
        shift_res (int): shift residues by this value
        outputFileName (str): realpath to modified score file
            if "PDBID_mod.score" is left as default: try to automatically detect
            pattern based on score_fin and add the "_mod" part into filename.
        res_cols (tuple/list): columns with residue numbers
        score_col (tuple/list): column with score/confidence

    Returns:
        outputFileName (str): realpath to modified score file
    """
    resi, resj = _misc.read_file(score_fin, usecols=res_cols, skiprows='auto', dtype=np.int_)
    score = _misc.read_file(score_fin, usecols=score_col, skiprows='auto', dtype=np.float_)

    # shift residues
    print(f"Shifting score file residues by {shift_res}.")
    resi += shift_res
    resj += shift_res

    if outputFileName == "PDBID_mod.score":
        dirpath = _misc.dirpath(score_fin)
        base = _misc.get_base(score_fin)
        ext = _misc.get_extension(score_fin)
        outputFileName = f"{dirpath}/{base}_mod{ext}"

    with open(outputFileName, "w") as fout:
        for i in range(len(resi)):
            fout.write(f"{resi[i]}\t{resj[i]}\t{score[i]}\n")

    outputFileName = _misc.realpath(outputFileName)
    print(f"Saved file as: {outputFileName}")
    return outputFileName


def DCAREX_modify_topology(top_fin, DCA_used_fin, force_k=10, DCA_skiprows="auto", outputFileName="PDBID_topol_mod.top"):
    """
    Modify topology:
        - topology file: use as template
        - used_DCA.txt: use all contacts as restraints
        - add all contacts with force constant force_k into bonds section of new topology

    Args:
        top_fin (str): topology file (path)
        DCA_used_fin (str): DCA used file (path)
        force_k (int/float): force constant in topology
        DCA_skiprows (int): ignore header rows of DCA_used_fin
            -1 or "auto": auto detect
        outputFileName (str):
            if "PDBID" in outputFileName: detect PDBID based on DCA_used_fin path
    """
    ATOM_I, ATOM_J = _misc.read_file(fin=DCA_used_fin, usecols=(2, 3), skiprows=DCA_skiprows, dtype=np.int_)

    # get n_DCA from DCA_used_fin
    with open(DCA_used_fin, "r") as read:
        WORDS = read.readline().split()
        for item in WORDS:
            if "top" in item:
                n_DCA = item[3:]

    # lstrip ./ or / from outputFileName
    if "/" in outputFileName:
        outputFileName = outputFileName.lstrip("/")
        outputFileName = outputFileName.lstrip("./")
    if "PDBID" in outputFileName:
        pdbid = _misc.get_PDBid(DCA_used_fin)
        outputFileName = pdbid + "_topol_mod.top"

    # read and write files
    new_dir = _misc.mkdir('logs')
    output = new_dir + "/" + outputFileName
    with open(top_fin, "r") as fin, open(output, "w") as fout:

        for line in fin:
            # default case: copy paste topology file
            fout.write(line)

            # add DCA bonds into bonds section
            if "[ bonds ]" in line:
                fout.write(f"; DCA bonds top{n_DCA}\n")
                fout.write(";  ai    aj funct            c0            c1            c2            c3\n")
                for i in range(len(ATOM_I)):
                    if type(force_k) is int:
                        fout.write("{:5d} {:5d} {:5d} {:13d} {:13d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, force_k))
                    elif type(force_k) is float or str:
                        if _misc.get_float_precision(force_k) == -1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.d}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, int(force_k)))
                        if _misc.get_float_precision(force_k) == 1:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.1f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(force_k)))
                        else:
                            fout.write("{:5d} {:5d} {:5d} {:13d} {:13.2f}\n".format(ATOM_I[i], ATOM_J[i], 9, 0, float(force_k)))
                fout.write("; native bonds\n")

    print("Saved modified topology as:", output)
    return


def DCAREX_modify_topology_v2(top_fin, temp_fin, force_range=[10, 40], lambda_scaling="T", outputFileName="PDBID_topol_mod.top"):
    """
    TODO

    Modify topology with lambda scaling prop to temperature.

    Args:
        top_fin
        temps_fin
        lambda_scaling:
            T: prop. to T
          1/T: prop to 1/T
    """
    # 1) calculate lambdas prop/antiprop to temperatures
    with open(temp_fin, "r") as fin:
        for line in fin:
            TEMPS = line
        TEMPS = TEMPS.split(", ")
        TEMPS = [float(item) for item in TEMPS[:-1]]

        T_min = TEMPS[0]
        T_max = TEMPS[-1]

        LAMBDA = []
        for T in TEMPS:
            lambda_i = (T-T_min)/(T_max-T_min)
            LAMBDA.append(round(lambda_i, 3))
        if lambda_scaling == "1/T":
            LAMBDA = [(1-lambda_i) for lambda_i in LAMBDA]

        FORCE = []
        force_min = force_range[0]
        force_max = force_range[-1]
        force_delta = force_max - force_min
        for lambda_i in LAMBDA:
            FORCE.append(force_min + lambda_i*force_delta)

    # 2) write modified topology for each replica with scaling lambda
    # lstrip ./ or / from outputFileName
    if "/" in outputFileName:
        outputFileName = outputFileName.lstrip("/")
        outputFileName = outputFileName.lstrip("./")
    if "PDBID" in outputFileName:
        pdbid = _misc.get_PDBid(top_fin)
        new_dir = _misc.mkdir('logs')

    rex_label = 0
    for force in FORCE:
        rex_label += 1
        # read and write files
        outputFileName = pdbid + f"_topol_mod_rex{rex_label}.top"
        output = new_dir + "/" + outputFileName
        with open(top_fin, "r") as fin, open(output, "w") as fout:

            modify_section = False
            for line in fin:
                # find DCA bonds section
                if "; DCA bonds" in line:
                    modify_section = True
                elif "; native bonds" in line:
                    modify_section = False
                    print("Saved modified topology as:", output)

                if modify_section == False:
                    fout.write(line)  # default case: copy paste
                else:
                    if ";" in line:
                        fout.write(line)
                    elif ";" not in line:
                        l = line.split()
                        l = [int(item) for item in l]
                        fout.write("{:5d} {:5d} {:5d} {:13d} {:13.3f}\n".format(l[0], l[1], l[2], l[3], force))

    return
