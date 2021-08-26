# @Author: Arthur Voronin <arthur>
# @Date:   26.08.2021
# @Filename: func.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

"""
.. hint:: This module is a collection of miscellaneous functions which are used
    frequently. Included functions may contain modified versions of small existing
    functions to extend their default behavior in order to streamline pyrexMD.
"""


import numpy as np
import os
import glob
import shutil
import termcolor

from pyrexMD.misc.classes import Error, dtypeError


################################################################################
################################################################################
### os-like modifications


def get_filename(path, search_file=True):
    """
    get filename (removes prefix of realpath of <path>).

    Args:
        path (str)
        search_file (bool):
          | True: use <path> as search pattern to search for existing file
          | -> remove prefix if only 1 match found to obtain existing filename
          | False: just remove prefix of given <path>

    Returns
        filename (str)
    """
    if search_file:
        file_list = glob.glob(path)
        if len(file_list) >= 1:
            path = glob.glob(path)[0]

    real_path = os.path.realpath(path)
    dir_path = os.path.dirname(os.path.realpath(path))
    filename = real_path[len(dir_path)+1:]
    return filename


def joinpath(filedir, filename, create_dir=True, realpath=True):
    """
    Returns joined path of (filedir + filename).

    Special case:
        filename contains relative or absolute path: ignore filedir input

    Reason:
        intention is to save/load file under the path "filename"


    Args:
        filedir (str): relative or absolute path of file directory
        filename (str): filename or relative/absolute filepath
        create_dir (bool): create filedir (w/o creating parent directories)
        realpath (bool): return realpath

    Returns:
        file_path (str)
    """
    if "/" in filename:
        if realpath:
            file_path = os.path.realpath(os.path.join(filedir, os.path.realpath(filename)))
        else:
            file_path = os.path.relpath(os.path.join(filedir, os.path.realpath(filename)))
    else:
        if realpath:
            file_path = os.path.realpath(os.path.join(filedir, filename))
        else:
            file_path = os.path.relpath(os.path.join(filedir, filename))
    if create_dir:
        # filedir: passed file directory
        # dirname: dirname of file_path after joining (filedir + filename)
        #          SPECIAL CASE: filedir ignored if filename contains rel. or abs. path
        #          -> dirname is not always equal to filedir
        dirname = os.path.dirname(os.path.realpath(file_path))
        mkdir(dirname)

    return file_path


def mkdir(path, verbose=True):
    """
    Make directory.

    Args:
        path (str): directory path
        verbose (bool): print message ('New folder created: ...')

    Returns:
        realpath (str)
            realpath to new directory

    Example:
        | >> cwd = /home/user/
        | >> mkdir('Test1')
        | New folder created: /home/user/Test1
        | >>mkdir('../Test2')
        | New folder created: /home/Test2
    """
    realpath = os.path.realpath(path)
    if not os.path.exists(realpath):
        os.makedirs(realpath)
        if verbose:
            cprint(f"New folder created: {realpath}", "blue")
    return realpath


def cd(path, verbose=True):
    """
    Change directory.

    Args:
        path (str): directory path
        verbose (bool): print message ('Changed directory to: ...')

    Returns:
        realpath (str)
            realpath of changed directory
    """
    realpath = os.path.realpath(path)
    os.chdir(realpath)
    if verbose:
        cprint(f"Changed directory to: {realpath}", "blue")
    return realpath


def cp(source, target, create_dir=True, verbose=True):
    """
    Copy file(s) from <source> to <target>. Will overwrite existing files and directories.

    Args:
        source (str, list, array): source path or list of source paths
        target (str): target path
        create_dir (bool)
        verbose (bool)

    Returns:
        target (str)
            target realpath
    """
    if not isinstance(target, str):
        raise dtypeError("Wrong datatype: <target> must be str (path/realpath).")
    if "*" in source:
        source = glob.glob(source)
    target = os.path.realpath(target)

    if isinstance(source, str):
        if os.path.isdir(source):
            shutil.copytree(source, target, dirs_exist_ok=True)
            if verbose:
                cprint(f"Copied source files to: {target}", "blue")
        else:
            shutil.copy(source, target)
            if verbose:
                cprint(f"Copied source file to: {target}", "blue")
    elif isinstance(source, list):
        if create_dir:
            mkdir(target, verbose=False)
        for item in source:
            if os.path.isdir(item):
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy(item, target)
        if verbose:
            cprint(f"Copied source files to: {target}", "blue")
    else:
        raise dtypeError("source", "str or list of str")
    return target


def rm(path, pattern=None, verbose=True):
    """
    Alias function of remove(). Remove file(s) from path.

    .. Note:: pattern can be implemented into path variable directly.

    Args:
        path (str): directory path
        pattern (None, str):
          | pattern
          | None: check for files with path only
          | str:  check for files with joined path of path + pattern
        verbose (bool): print message ('removed file: ... ')
    """
    if pattern is None:
        realpath = os.path.realpath(path)
    else:
        realpath = joinpath(path, pattern)

    # remove files
    for item in glob.glob(realpath):
        os.remove(item)
        if verbose:
            cprint(f"removed file: {item}", "red")
    return


################################################################################
################################################################################
### misc

def cprint(msg, color=None, on_color=None, attr=None, **kwargs):
    """
    Modified function of termcolor.cprint(). Apply colored print.

    Args:
        msg (str)
        color (None, str): "grey", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
        on_color (None, str): "on_grey", "on_red", "on_green", "on_yellow", "on_blue", "on_magenta", "on_cyan", "on_white"
        attr (None, list):
          | attributes list
          | bold, dark, underline, blink, reverse, concealed

    Keyword Args:
        cprint_color (None, str): alias for color (with higher priority)
        sep (str): string inserted between values, default a space.
        end (str): string appended after the last value, default a newline.
        file (str): a file-like object (stream); defaults to the current sys.stdout.
        flush (bool): whether to forcibly flush the stream.
    """
    if "cprint_color" in kwargs:
        color = kwargs["cprint_color"]
        del kwargs["cprint_color"]
    termcolor.cprint(msg, color, on_color, attrs=attr, **kwargs)
    return


def percent(num, div, prec=2):
    """
    Returns the percentage of num/div as float

    Args:
        num (int): numerator
        div (int): divisor
        prec (None, int): rounding precision

    Returns:
        p (float)
            p = 100 * num/div
    """
    num = float(num)
    div = float(div)
    if div == 0:
        return 0.0             # no division by zero
    p = round(100 * num / div, prec)
    return p


def round_down(number, base=1):
    """
    Args:
        number (float/int)
        base (float/int)

    Example:
      | for i in range(11):
      |     print(i, round_down(i,5))
      | (0, 0)
      | (1, 0)
      | (2, 0)
      | (3, 0)
      | (4, 0)
      | (5, 5)   # does not round down if remainder is zero
      | (6, 5)
      | (7, 5)
      | (8, 5)
      | (9, 5)
      | (10, 10)
    """
    return number - (number % base)


def round_up(number, base=1):
    """
    Args:
        number (float/int)
        base (float/int)

    Example:
      | for i in range(11):
      |     print(i, round_up(i,5))
      | (0, 0)
      | (1, 5)
      | (2, 5)
      | (3, 5)
      | (4, 5)
      | (5, 5)   # does not round up if remainder is zero
      | (6, 10)
      | (7, 10)
      | (8, 10)
      | (9, 10)
      | (10, 10)
    """
    if (number % base) == 0:
        return number
    else:
        return number + base - (number % base)


def round_to_base(number, base=1):
    """
    Round up or down depending on smallest difference to base.

    Args:
        number (float/int)
        base (float/int)
    """
    up = round_up(number, base)
    down = round_down(number, base)
    if abs(up-number) < abs(number-down):
        return up
    else:
        return down


def round_object(object, prec=3, dtype=None):
    """
    Args:
        object (list, array):
          | single list/array
          | list of lists/arrays
          | array of lists/arrays
        prec (int): rounding precision
        dtype (None, dtype): return object values as specific data type

    Returns:
        new_object (list, array)
            same structure as object, but with rounded values.
    """
    # single list/array (if error occurs, maybe add more np.<data types> to this case)
    if isinstance(object[0], (int, float, np.float32, np.int32)):
        new_object = [round(item, prec) for item in object]

    # list of lists
    elif isinstance(object[0], list):
        new_object = []
        for sublist in object:
            if isinstance(sublist[0], (int, float)):
                new_sublist = [round(item, prec) for item in sublist]
                new_object.append(new_sublist)
            else:
                new_object.append(sublist)

    # list of arrays
    elif isinstance(object[0], np.ndarray):
        new_object = []
        for subarray in object:
            new_subarray = np.around(subarray, prec)
            new_object.append(new_subarray)

    # array of lists/arrays: convert outer list to array
    if isinstance(object, np.ndarray):
        new_object = np.array(new_object, dtype=dtype)

    return new_object


def get_substrings(string, seperators=["/", "_", ".", ":", ";", " "], reverse=False):
    """
    get substrings of string by removing each seperator.

    Args:
        string (str)
        seperators (list)
        reverse (bool): return substrings in reversed order

    Returns:
        substrings (list)
            list of substrings

    Example:
        | >> get_substrings("split/this_string.into:parts")
        | ['split', 'this', 'string', 'into', 'parts']
        | >> get_substrings("split/this_string.into:parts", reverse=True)
        | ['parts', 'into', 'string', 'this', 'split']
    """
    for sep in seperators:
        string = string.replace(sep, " ")
    substrings = [item for item in string.split(" ") if len(item) > 0]
    if reverse:
        substrings = substrings[::-1]
    return substrings


def split_lists(A, n, remap=False):
    """
    split list A into two lists B1 and B2 according to A = B1 + B2,
    where B2 contains every n-th element of A.

    Args:
        A (int, list, array):
          | int: initialize A according to: A = list(range(A))
          | list, array: input list A which should be split
        n (int): split every n-th element of A into B2
        remap (bool): remap Bi according to: Bi = range(0, len(Bi))

    Returns:
        A (list)
            input list A
        B1 (list)
            partial list B1
        B2 (list)
            partial list B2 (contains every n-th element of A)
    """
    if isinstance(A, int):
        A = list(range(A))
    elif isinstance(A, (type(range(0)), type((i for i in range(0))))):
        A = list(A)
    elif not isinstance(A, (list, np.ndarray)):
        raise TypeError("A has to be int, list/np.array or range/generator.")

    B1 = [i for k, i in enumerate(A) if (k+1) % n != 0]
    B2 = [i for k, i in enumerate(A) if (k+1) % n == 0]

    if remap:
        B1 = list(range(len(B1)))
        B2 = list(range(len(B2)))
    return A, B1, B2


def flatten_array(x):
    """
    Returns flattened array of x.

    Args:
        x (array, array of arrays)

    Returns:
        flattened_array (array)
    """
    try:
        flattened_array = [item for sub in x for item in sub]
    except TypeError:
        flattened_array = x
    return flattened_array


def norm_array(array, start_value=0):
    """
    norm array

      - starts with <start_value>
      - deltaValue is fixed to 1 (i.e. array[i]- array[i+1] = 1)

    Args:
        array (list, array)

    Returns:
        normed_array (list, array)
    """
    l = len(array)
    normed_array = [0]*l
    if isinstance(array, np.ndarray):
        normed_array = np.array(normed_array)
    for i in range(l):
        normed_array[i] = i + start_value
    return normed_array


def get_precision(number):
    """
    Returns leading precision of number.

      - convert float to string
      - find position of decimal point relative to the length of the string

    Args:
        number (int, float)

    Returns:
        precision (int)

    Example:
        | >> get_precision(5.00000)
        | 1
        | >> get_precision(5.12300)
        | 3
    """
    if type(number) is str:
        number = number.rstrip("0")
    s = str(number)
    precision = s[::-1].find('.')
    return precision


def get_base(_str):
    """
    Get base (i.e. everything except extension) of string "_str".

    Args:
        _str (str)

    Returns:
        base (str)
            base of _str

    Example:
        | >> get_base("random_plot.png")
        | "random_plot"
    """
    _str = str(_str).rsplit("/", 1)[-1]
    base = _str.rsplit(".", 1)[0]
    return base


def get_extension(_str):
    """
    Get ".ext" (extension) of string "_str".

    Args:
        _str (str)

    Returns:
        ext (str)
            extension of _str

    Example:
        | >> get_extension("random_plot.png")
        | ".png"
    """
    _str = str(_str).rsplit("/", 1)[-1]
    ext = "."+_str.rsplit(".", 1)[-1]
    return ext


def insert_str(str1, str2, sep, loc="after"):
    """
    Insert str2 in str1.

    Args:
        str1 (str): base string
        str2 (str): append string
        sep (str): seperator
        loc (str): insert location: 'before' or 'after' first seperator in str1

    Returns:
        _str (str)

            new string
    """
    if loc == "after":
        _str = str1.split(sep)[0] + sep + str2 + f"{sep}".join(str1.split(sep)[1:])
    elif loc == "before":
        _str = str1.split(sep)[0] + str2 + sep + f"{sep}".join(str1.split(sep)[1:])
    else:
        raise Error("loc argument must be either 'before' or 'after'.")
    return _str


def autodetect_header(fin):
    """
    Autodetects header.

    Args:
        fin (str): input file (path)

    Returns:
        header_rows (int)
    """
    header_rows = 0
    extension = get_extension(fin)
    CHAR_SET = set("\n\t .,0123456789")

    with open(fin, "r") as f:
        lines = f.readlines()
        n_lines = len(lines)

        for line in lines:
            if extension == ".pdb":
                l = "".join(line.split())
                if l.isnumeric():
                    break
                if "ATOM" in l[0:4]:
                    break
                else:
                    header_rows += 1

            else:
                l = set(line)
                if l.issubset(CHAR_SET):
                    break
                else:
                    header_rows += 1

        # check if header was found
        if n_lines == header_rows:
            header_rows = 0
    return header_rows


def read_file(fin, sep=None, usecols=(0, 1), n_rows=None, skiprows='auto', dtype=float):
    """
    Read file and return tuple of np.arrays with data for specified columns.

    Args:
        fin (str): input file (path)
        sep (None, str):
          | seperator of columns.
          | None: whitespace
        usecols (int, sequence): data columns
        n_rows (None, int): lenght of returned data (AFTER skipping rows)
        skiprows (None, 'auto', int):
          | ignore header rows of fin
          | 'auto' or -1: auto detect
        dtype (dtype cls, list, tuple):
          | data type of returned np.array
          | dtype cls: all data types are same
          | list/tuple of dtype cls: specify dtype of each data column
          |
          | valid dtypes:
          |     str, int, float, complex

    Returns:
        data_array (array)
            if usecols is int ~ data of one column specified in usecols
        data_list (list)
            if usecols is list ~ data of each column specified in usecols

    Example:
        | # single columns per fin
        | X = read_file('data.txt', usecols=0)
        | Y = read_file('data.txt', usecols=1)
        | # multiple columns per fin
        | X,Y = read_file('data.txt', usecols=(0,1))
    """
    if skiprows == "auto" or skiprows == -1:
        skiprows = autodetect_header(fin)
    if skiprows == None:
        skiprows = 0

    if type(usecols) == int:
        col = np.loadtxt(fin, delimiter=sep, skiprows=skiprows, usecols=usecols, dtype=dtype)
        col = col[:n_rows]
        return col
    else:
        DATA = []
        for i in range(len(usecols)):
            if isinstance(dtype, (list, tuple)):
                col = np.loadtxt(fin, delimiter=sep, skiprows=skiprows, usecols=usecols[i], dtype=dtype[i])
                col = col[:n_rows]
            else:
                col = np.loadtxt(fin, delimiter=sep, skiprows=skiprows, usecols=usecols[i], dtype=dtype)
                col = col[:n_rows]
            DATA.append(col)
        return DATA


def read_DCA_file(DCA_fin, n_DCA, usecols=(0, 1), skiprows='auto', filter_DCA=True, RES_range=[None, None]):
    """
    Read DCA file and return DCA pairs IJ.

    Args:
        DCA_fin (str): DCA input file (path)
        n_DCA (int): number of used DCA contacts
        usecols (tuple, list): columns containing the RES pairs in DCA_fin
        skiprows ('auto', int):
          | ignore header rows of DCA_fin
          | 'auto' or -1: auto detect
        filter_DCA (bool):
          | True: ignore DCA pairs with abs(i-j) < 4
          | False: use all DCA pairs w/o applying filter
        RES_range (None):
          | Use RES within [RES_min, RES_max] range.
          | None: do not narrow down RES range

    Returns:
        DCA_PAIRS (list)
            DCA pairs IJ
        DCA_PAIRS_RES_range (list)
            [RES_min, RES_max] of DCA_PAIRS (not input RES_range values)
    """
    I, J = read_file(DCA_fin, usecols=usecols, skiprows=skiprows, dtype=np.int_)
    DCA_PAIRS = []
    if RES_range[0] is None:
        RES_range[0] = min(min(I), min(J))
    if RES_range[1] is None:
        RES_range[1] = max(max(I), max(J))

    for k in range(len(I)):
        if len(DCA_PAIRS) == n_DCA:
            break
        if filter_DCA:
            if abs(I[k] - J[k]) < 4:
                pass    # ignore sites with indexdiff < 4
            else:
                if I[k] >= RES_range[0] and J[k] <= RES_range[1]:
                    DCA_PAIRS.append((I[k], J[k]))
                else:
                    pass
        else:
            DCA_PAIRS.append((I[k], J[k]))

        DCA_PAIRS_RES_range = [min(min(I), min(J)), max(max(I), max(J))]
    return DCA_PAIRS, DCA_PAIRS_RES_range


def get_PDBid(ref):
    """
    Obtains PDB id from ref by splitting its path and searching for alpha numerical (alnum) strings with length 4.

    Args:
        ref (str, universe, atomgrp): reference structure

    Returns:
        PDBid (str)
            if ref path contains one unique alnum string with length 4
        PDBid (list)
            if ref path contains multiple unique alnum strings with length 4
        None (None)
            else
    """
    if isinstance(ref, str):
        ref = ref.replace('.', '/')
        ref = ref.replace('_', '/')
        split = [i.lower() for i in ref.split('/')]

    else:
        try:
            ref = ref.universe.filename
            ref = ref.replace('.', '/')
            ref = ref.replace('_', '/')
            split = [i.lower() for i in ref.split('/')]

        except AttributeError:
            print("get_PDBid(ref): AttributeError\
                 \nCan't obtain PDBid since ref is neither a path nor a MDA universe/atomgrp with the attribute ref.universe.filename")
            return

    PDBid = []   # list for guessed PDBids
    split = [i for i in split if i not in ["logs", "used", "from", "with", "home"]]   # filter guessed PDBids

    # search algorithm
    for item in split:
        if len(item) == 4 and item.isalnum() and item not in PDBid:
            PDBid.append(item)
    if len(PDBid) == 1:
        return PDBid[0]
    elif len(PDBid) > 1:
        print("get_PDBid(ref): Warning\
             \nFound multiple possible PDB ids by splitting the ref path.")
        print(PDBid)
        PDBid = input('Enter correct PDB id:\n')
        return PDBid
    else:
        print("get_PDBid(ref): Warning\
             \nNo PDB id was found by splitting the ref path.")
        return


def print_table(data=[], prec=3, spacing=8, dtype=None, verbose=True, verbose_stop=30, warn=True, **kwargs):
    """
    Prints each item of "data" elementwise next to each other as a table.

    Args:
        data (list of lists): table content, where each list corresponds to one column
        prec (None, int):
          | None: rounding off
          | int:  rounding on
        spacing (int):
          | spacing between columns (via "".expandtabs(spacing))
          | 8: default \\t width
        dtype (None, dtype): dtype of table values after rounding
        verbose (bool): print table
        verbose_stop (None, int): stop printing after N lines.
        warn (bool): print/supress message  "misc.print_table(): printed only N entries (set by verbose_stop parameter)."

    Returns:
        table_str (str)
            table formatted string. Can be used to save log files.

    Example:
        | letters = ["A", "B", "C"]
        | numbers = [1, 2, 3]
        | table = misc.print_table([letters, numbers])
        | >>  A   1
        |     B   2
        |     C   3
        |
        | # save table as log file:
        | with open("table.log", "w") as fout:
        |     fout.write(table)
    """
    # apply precision to input
    if prec is not None:
        data = round_object(data, prec=prec)

    if dtype is not None:
        data = np.array(data, dtype=dtype)

    table_str = ""

    if len(data) == 1:
        for item in data:
            table_str += "{}\n".format(item)
    elif len(data) == 2:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\n".format(data[0][ndx], data[1][ndx])
    elif len(data) == 3:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                               data[2][ndx])
    elif len(data) == 4:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                   data[2][ndx], data[3][ndx])
    elif len(data) == 5:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                       data[2][ndx], data[3][ndx],
                                                       data[4][ndx])
    elif len(data) == 6:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                           data[2][ndx], data[3][ndx],
                                                           data[4][ndx], data[5][ndx])
    elif len(data) == 7:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                               data[2][ndx], data[3][ndx],
                                                               data[4][ndx], data[5][ndx],
                                                               data[6][ndx])
    elif len(data) == 8:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                                   data[2][ndx], data[3][ndx],
                                                                   data[4][ndx], data[5][ndx],
                                                                   data[6][ndx], data[7][ndx])
    elif len(data) == 9:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                                       data[2][ndx], data[3][ndx],
                                                                       data[4][ndx], data[5][ndx],
                                                                       data[6][ndx], data[7][ndx],
                                                                       data[8][ndx])
    elif len(data) == 10:
        for ndx, item in enumerate(data[0]):
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
                                                                           data[2][ndx], data[3][ndx],
                                                                           data[4][ndx], data[5][ndx],
                                                                           data[6][ndx], data[7][ndx],
                                                                           data[8][ndx], data[9][ndx])
    elif len(data) > 10:
        raise Error(f"Extend code of {__name__}.{print_table.__name__}() to work with more than 10 elements.")

    table_str = table_str.expandtabs(spacing)

    if verbose:
        for item in table_str.splitlines()[:verbose_stop]:
            print(item)
        if verbose_stop is not None:
            if (len(table_str.splitlines()) >= verbose_stop):
                if warn:
                    cprint(f"misc.print_table(): printed only {verbose_stop} entries (set by verbose_stop parameter).", "blue")
    return table_str


def save_table(data=[], filename="", header="", default_dir="./logs", prec=3, verbose=True, verbose_stop=None, **kwargs):
    """
    Executes misc.print_table() and saves the returned table string as a log file.

    Args:
        data (list of lists): table content, where each list corresponds to one column
        filename (str): filename or realpath
        header (str): header for log file, e.g. column descriptions
        default_dir (str)
        prec (None, int):
          | None: rounding off
          | int:  rounding on
        verbose (bool)
        verbose_stop (None, int)

    Keyword Args:
        save_as (str): alias of filename

    Return:
        realpath (str)
            path to log file
    """
    if "save_as" in kwargs:
        filename = kwargs["save_as"]
    if filename == "":
        raise TypeError("misc.save_table(): missing 1 required argument: 'filename' or 'save_as'")

    ext = get_extension(filename)
    if ext != ".log":
        filename += ".log"
    realpath = joinpath(default_dir, filename, create_dir=True)

    with open(realpath, "w") as fout:
        if header != "":
            fout.write(f"{header}\n")

        ref_shape = np.shape(data[0])
        for item in data:
            if np.shape(item) != ref_shape:
                raise IndexError('save_table(): data consists of lists with unequal lengths.')

        table_str = print_table(data, prec, verbose=False, verbose_stop=verbose_stop, **kwargs)
        fout.write(table_str)

    if verbose:
        cprint(f"Saved logfile as: {realpath}", "blue")
    return realpath


def get_slice_indices(list1, list2):
    """
    Get slice indices of two lists (mainlist and sublist) where sublist is a
    slice of mainlist, e.g:
    sublist = mainlist[slice_ndx1:slice_ndx2]

    Args:
        list1 (list, array): mainlist or sublist
        list2 (list, array): mainlist or sublist

    Returns:
        slice_ndx1 (int)
            first slice index ("start")
        slice_ndx2 (int)
            second slice index ("end")
    """
    if len(list1) > len(list2):
        mainlist = list1
        sublist = list2
    else:
        mainlist = list2
        sublist = list1

    # find potential starting indices
    test = np.array(mainlist)
    test_ndx = np.where(test == sublist[0])[0]

    for ndx in test_ndx:
        if sublist == mainlist[ndx:ndx+len(sublist)]:
            slice_ndx1 = ndx
            slice_ndx2 = ndx+len(sublist)
            return (slice_ndx1, slice_ndx2)
    raise Error("There is no sublist between list1 and list2. Slicing is not possible.")


def get_cutoff_array(array, cut_min=None, cut_max=None):
    """
    Applies cutoff on array and returns array with values within cutoff range
    as well as the corresponding indices.

    Args:
        array (array/list)
        cut_min (None, int, float): min value for cutoff
        cut_max (None, int, float): max value for cutoff

    Returns:
        cutoff_array (array, list)
            array with values within cutoff
        cutoff_array_ndx (array, list)
            coresponding indices to map correctly

    Example:
        | array = list(range(10,20))
        | cut_min, cut_max = 12, 16
        | cutoff_array, cutoff_array_ndx = misc.get_cutoff_array(array, cut_min, cut_max)
        | print(cutoff_array)
        | >> [12, 13, 14, 15, 16]
        | print(cutoff_array_ndx)
        | >> [2, 3, 4, 5, 6]
    """
    # no cutoff
    if cut_min is None and cut_max is None:
        cutoff_array = array
        cutoff_array_ndx = [ndx for ndx, item in enumerate(array)]

    # with cutoff
    else:
        if cut_min is None:
            cut_min = min(array)
        if cut_max is None:
            cut_max = max(array)
        cutoff_array = [item for item in array if item >= cut_min if item <= cut_max]
        cutoff_array_ndx = [ndx for ndx, item in enumerate(array) if item >= cut_min if item <= cut_max]

    # convert output to array if input was array
    if isinstance(array, np.ndarray):
        cutoff_array = np.array(cutoff_array)
        cutoff_array_ndx = np.array(cutoff_array_ndx)
    return (cutoff_array, cutoff_array_ndx)


def get_subarray_start_ndx(array, subarray):
    """
    Returns start index of <array> at which <subarray> matches.

    Args:
        array (list, array)
        subarray (list, arr)

    Returns:
        start_ndx (None, int)
          | None: arrays do not match
          | int: starting index of <array> at which <subarray> matches
    """
    if len(array) < len(subarray):
        raise ValueError("get_subarray_start(array, subarray): lenght(array) < length(subarray).")
    if isinstance(array, np.ndarray):
        array = array.tolist()
    if isinstance(subarray, np.ndarray):
        subarray = subarray.tolist()

    ndx_list = [x for x in range(len(array)) if array[x:x+len(subarray)] == subarray]
    if len(ndx_list) == 0:
        start_ndx = None
    else:
        start_ndx = ndx_list[0]
    return start_ndx


def get_subarray(array, ndx_sel):
    """
    Returns subarray of <array> for items with indices <ndx_sel>.

    Args:
        array (array, list): array-like object
        ndx_sel (array, list): index selection

    Returns:
        subarray (array)
            subarray of array
    """
    subarray = []
    for ndx in ndx_sel:
        subarray.append(array[ndx])
    return np.array(subarray)


def get_sorted_array(array, reverse=False, verbose=True, **kwargs):
    """
    Returns sorted array (increasing order) and the coresponding element indices
    of the input array.

    Args:
        array (array, list): array-like object

    Returns:
        SORTED_VALUES (array)
            sorted array
        SORTED_NDX (array)
            corresponding indices
        reverse (bool):
          | False: ascending ranking order (low to high)
          | True:  decending ranking order (high to low)
        verbose (bool): print table with RANKED_VALUES, RANKED_NDX

    Keyword Args:
        prec(None, int):
          | None: rounding off
          | int:  rounding on
        spacing (int):
          | spacing between columns (via "".expandtabs(spacing))
          | 8: default \\t width
        verbose_stop (None, int): stop printing after N lines

    Example:
        | >> A = np.array([1, 3, 2, 4])
        | >> get_sorted_array(A)
        | (array([1, 2, 3, 4]), array([0, 2, 1, 3]))
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if not reverse:
        SORTED_ARRAY = np.sort(array)
        SORTED_NDX = np.argsort(array)
    else:
        SORTED_ARRAY = np.flip(np.sort(array))
        SORTED_NDX = np.flip(np.argsort(array))

    if verbose:
        print_table([SORTED_ARRAY, SORTED_NDX], **kwargs)

    return (SORTED_ARRAY, SORTED_NDX)


def get_ranked_array(array, reverse=False, verbose=True, **kwargs):
    """
    Returns ranked array (decreasing order) and the corresponding element indices of the input array.

    Args:
        array (array, list): array-like object
        reverse (bool):
          | False: decending ranking order (high to low)
          | True:  ascending ranking order (low to high)
        verbose (bool): print table with RANKED_VALUES, RANKED_NDX

    Keyword Args:
        prec(None, int):
          | None: rounding off
          | int:  rounding on
        spacing (int):
          | spacing between columns (via "".expandtabs(spacing))
          | 8: default \\t width
        verbose_stop (None, int): stop printing after N lines

    Returns:
        RANKED_VALUES (array)
            ranked values
        RANKED_NDX (array)
            corresponding indices

    Example:
        | >> A = np.array([1, 3, 2, 4])
        | >> get_ranked_array(A, verbose=False)
        | (array([4, 3, 2, 1]), array([3, 1, 2, 0]))
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if reverse:
        RANKED_ARRAY = np.sort(array)
        RANKED_NDX = np.argsort(array)
    else:
        RANKED_ARRAY = np.flip(np.sort(array))
        RANKED_NDX = np.flip(np.argsort(array))

    if verbose:
        print_table([RANKED_ARRAY, RANKED_NDX], **kwargs)
    return (RANKED_ARRAY, RANKED_NDX)


def get_quantile(data, p, prec=2, **kwargs):
    """
    Alias function of np.quantile(). Get value (or score) below which p% of the
    observations may be found.

    Args:
        data (array-like)
        p (int, float): percent value ~ range 0-1
        prec (None, int): rounding precision

    Returns:
        quantile (int, float)
    """
    quantile = round(np.quantile(data, p, **kwargs), prec)
    return quantile


def get_percentile(data, p, prec=2, **kwargs):
    """
    Alias function of np.percentile(). Get value (or score) below which p% of
    the observations may be found.

    Args:
        data (array-like)
        p (int, float): percent value ~ range 0-100
        prec (None, int): rounding precision

    Returns:
        percentile (int, float)
    """
    percentile = round(np.percentile(data, p, **kwargs), prec)
    return percentile
