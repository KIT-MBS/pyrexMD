# miscellaneous
from __future__ import division, print_function
import PIL  # pillow/PIL (Pyhon Imaging Library)
import pdf2image
import pickle as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import subprocess
import time
import copy


################################################################################
################################################################################
### CONFIG CLASS


class CONFIG(object):
    """
    Class to store config data.
    """

    def __init__(self, dict_=None, **kwargs):
        """
        Init config object by passing a dict, kwargs or both.
        """
        if dict_ != None:
            self.__dict__.update(dict_)
        self.__dict__.update(**kwargs)
        return

    def __call__(self):
        """
        Alias for print_config() method.
        """
        self.print_config()
        return

    def __getitem__(self, key):
        """
        Make class objects subscriptable, i.e. access data via [] ~ brackets.
        """
        return self.__dict__[str(key)]

    def deepcopy(self):
        """
        Return deep copy of object.
        """
        return copy.deepcopy(self)

    def keys(self):
        """
        Return dict keys.
        """
        return self.__dict__.keys()

    def values(self):
        """
        Return dict values.
        """
        return self.__dict__.values()

    def items(self):
        """
        Return dict items (key-value pairs).
        """
        return self.__dict__.items()

    def update_config(self, dict_=None, **kwargs):
        """
        Update config object by passing a dict, kwargs or both.
        """
        if dict_ != None:
            self.__dict__.update(dict_)
        self.__dict__.update(**kwargs)
        return

    def update_by_alias(self, alias="", key="", key_ndx=None, **kwargs):
        """
        Update key value for an alias if alias is in kwargs.

        Detailed Example:
            default = {"colors": ["g", "r"],
                       "ms": 1}
            alias_dict = {"color_positive": "teal",
                          "color_negative": "orange",
                          "markersize": 5}
            cfg = misc.CONFIG(default, **alias_dict)
            cfg.print_config()
            >>  key                 value
                colors              ['g', 'r']
                ms                  1
                color_positive      teal
                color_negative      pink
                markersize          5

            cfg.update_by_alias(alias="markersize", key="ms", **alias_dict)
            cfg.update_by_alias(alias="color_positive", key="colors", key_ndx=0, **alias_dict)
            cfg.update_by_alias(alias="color_negative", key="colors", key_ndx=1, **alias_dict)
            cfg.print_config()
            >>  key                 value
                colors              ['teal', 'orange']
                ms                  5
                color_positive      teal
                color_negative      orange
                markersize          5
        """
        if alias in kwargs:
            if key_ndx is None:
                self.__dict__[key] = kwargs[alias]
            else:
                self.__dict__[key][key_ndx] = kwargs[alias]
        return

    def print_config(self):
        """
        Print and return stored data in table format with "key" and "value" columns.
        """
        str_ = f"{'key':20}{'value':20}\n\n"

        for item in self.__dict__.items():
            str_ += f"{item[0]:20}{item[1]}\n"
        print(str_)

        return str_


################################################################################
################################################################################
### timer stuff


class TIMER(object):
    def __init__(self):
        self.t0 = 0
        self.t1 = 0
        return

    def _get_time(self):
        return time.time()


def timeit(timer=None):
    """
    Test elapsed time of a process

    Example:
        timer1 = timeit()
        <code>
        timer1 = timerit(timer1)
    """
    if timer == None:
        timer = TIMER()
        timer.t0 = timer._get_time()

    elif timer.t0 != 0:
        timer.t1 = timer._get_time()

    if timer.t1 != 0:
        t_diff = timer.t1 - timer.t0
        print("Elapsed time:", t_diff, "s")

    return timer


################################################################################
################################################################################
### linux-like cmds ~ os package


def pwd(print=True):
    """
    Print working directory

    Args:
        print (bool)

    Returns:
        cwd (str): current working directory
    """
    if print:
        print(os.getcwd())
    return os.getcwd()


def realpath(path):
    """
    Get realpath

    Args:
        path (str)

    Returns:
        realpath (str)
    """
    return os.path.realpath(path)


def dirpath(path):
    """
    Get realpath of the last directory of <path>.

    Args:
        path (str)

    Returns:
        realpath (str)
    """
    return(os.path.dirname(os.path.realpath(path)))


def joinpath(filedir, filename, create_dir=True):
    """
    Returns truepath of (filedir + filename).

    Special cases:
        filename contains relative path: ignores filedir input
        filename contains absolute path: ignores filedir input

    Reason: intention is to save/load file under the path "filename"

    Args:
        filedir (str): relative or absolute path of file directory
        filename (str):
            filename
            relative/absolute path of file: ignores filedir input
        create_dir (bool): create filedir (w/o creating parent directories)

    Returns:
        file_realpath (str)
    """
    file_realpath = os.path.realpath(filename)

    if "/" in filename:
        file_realpath = os.path.join(os.path.realpath(filedir), os.path.realpath(filename))
    else:
        file_realpath = os.path.join(os.path.realpath(filedir), filename)

    if create_dir:
        # filedir: passed file directory
        # dirname: dirname of file_realpath after joining paths
        #          NOTE: ignores filedir if it contains rel. or abs. path
        #          -> filedir != dirname
        dirname = os.path.dirname(file_realpath)
        mkdir(dirname)

    return file_realpath


def mkdir(path, verbose=True):
    """
    Create a new folder/directory

    Args:
        path (str): directory path

    Returns:
        realpath (str): realpath to new directory

    Example:
        cwd = /home/user/
        mkdir('Test1')
        >> New folder created: /home/user/Test1
        mkdir('../Test2')
        >> New folder created: /home/Test2
    """
    realpath = os.path.realpath(path)
    if not os.path.exists(realpath):
        os.makedirs(realpath)
        print('New folder created: ', realpath)
    return realpath


def bash_cmd(cmd):
    """
    Execte any bash command via python.

    Args:
        cmd (str): bash cmd

    Example:
    bash_cmd("ls")
    >>
    bash_cmd("touch test")
    bash_cmd("ls")
    >> test
    """
    subprocess.check_call(cmd.split())
    return
################################################################################
################################################################################
### pillow/PIL stuff


def convert_image(fin, fout, format="auto", **kwargs):
    """
    Converts image type. Currently ONLY tested for:
        tif -> png
        pdf -> png

    Args:
        fin (str): file path for file in (including extension)
        fout (str): file path for file out (including extension)
        format (str): fout format/extension
            "auto"
            "png"
    Kwargs:
        dpi (tuple): dpi for each direction
        quality (int): compression quality:  0 to 100
    """
    default = {"dpi": (300, 300),
               "quality": 100}
    cfg = CONFIG(default, **kwargs)
    ###############################
    if "pdf" in fin:
        image = pdf2image.convert_from_path(fin)[0]
    else:
        image = PIL.Image.open(fin)
        image.convert(mode="RGB")
    # output format
    if format == "auto":
        format = get_extension(fout)[1:]
    image.save(fout, format=format, **cfg)
    return


def convert_multiple_images(folder_in, folder_out, format="png", **kwargs):
    """
    Args:
        folder_in (str):  file path for folder with input  images
        folder_out (str): file path for folder with output images
        format (str): fout format/extension
            "auto"
            "png"

    Kwargs:
        dpi (tuple): dpi for each direction
        quality (int): compression quality:  0 to 100
    """
    default = {"dpi": (300, 300),
               "quality": 100}
    cfg = CONFIG(default, **kwargs)
    ###############################
    images = [os.path.join(folder_in, file) for file in os.listdir(folder_in) if os.path.isfile(os.path.join(folder_in, file))]
    for i in images:
        name = get_base(i)
        name_out = f"{folder_out}/{name}.{format.lower()}"
        convert_image(fin=i, fout=name_out, format=format, **cfg)
    return


################################################################################
################################################################################
### misc


def unzip(iterable):
    """
    alias function for zip(*iterable)

    Args:
        iterable

    Returns:
        zip(*iterable)
    """
    return zip(*iterable)


def get_python_version():
    """
    Returns python version.
    """
    return sys.version_info[0]


def input_python_x(msg):
    """
    Uses raw_input(msg) in python 2.x or input(msg) in python 3.x

    Args:
        msg (str): promt message

    Returns:
        msg (str): user input message
    """
    if sys.version_info[0] == 2:
        return raw_input(msg + "\n")
    else:
        return input(msg + "\n")


def percent(num, div, prec=2):
    """
    Returns the percentage of num/div as float:
        p = 100 * num/div

    Args:
        num (int): numerator
        div (int): divisor
        prec (None/int): rounding precission

    Returns:
        p (float): num/div as percentage
    """
    num = float(num)
    div = float(div)
    if div == 0:
        return 0.0             # no division by zero
    p = round(100 * num / div, prec)
    return p


def round_down(number, base):
    """
    Args:
        number (int)
        base (int)

    Example:
        for i in range(11):
            print(i, round_down(i,5))
        >> (0, 0)
           (1, 0)
           (2, 0)
           (3, 0)
           (4, 0)
           (5, 5)   # does not round down if remainder is zero
           (6, 5)
           (7, 5)
           (8, 5)
           (9, 5)
           (10, 10)
    """
    return number - (number % base)


def round_up(number, base):
    """
    Args:
        number (int)
        base (int)

    Example:
        for i in range(11):
            print(i, round_up(i,5))
        >> (0, 0)
           (1, 5)
           (2, 5)
           (3, 5)
           (4, 5)
           (5, 5)   # does not round up if remainder is zero
           (6, 10)
           (7, 10)
           (8, 10)
           (9, 10)
           (10, 10)
    """
    if (number % base) == 0:
        return number
    else:
        return number + base - (number % base)


def round_object(object, prec=3):
    """
    Args:
        object (list/array): can be any of these:
              - single list/array
              - list of lists/arrays
              - array of lists/arrays
        prec (int): rounding precission

    Returns:
        new_object (list/array): same structure as object, but with rounded values.

    """
    # single list/array
    if isinstance(object[0], (int, float)):
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
        new_object = np.array(new_object)

    return new_object


def get_substrings(string, seperators=[".", "/", "_", " ", ":", ";"]):
    """
    get substrings of string by removing each seperator.

    Args:
        string (str)
        seperators (list)

    Returns:
        substrings (list): list of substrings

    Example:
        get_substrings("split.this/string_into:parts")
        >> ['split', 'this', 'string', 'into', 'parts']
    """
    for sep in seperators:
        string = string.replace(sep, " ")
    substrings = [item for item in string.split(" ") if len(item) > 0]
    return substrings


def split_lists(A, n, remap=False):
    """
    split list A into two lists B1 and B2 according to
        A = B1 + B2
    where B2 contains every n-th element of A.


    Args:
        A (int/list/np.array):
            (int): initialize A according to: A = list(range(A))
            (list/np.array): input list A which should be split
        n (int): split every n-th element of A into B2
        remap (bool): remap Bi according to: Bi = range(0, len(Bi))

    Returns:
       A (list): input list A
       B1 (list): partial list B1
       B2 (list): partial list B2 (contains every n-th element of A)
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
        x (array/array of arrays)

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
    normed array:
        - starts with <start_value>
        - deltaValue is fixed to 1 (i.e. array[i]- array[i+1] = 1)

    Args:
        array (list/array)

    Returns:
        normed_array (list/array)
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
    Returns leading precission of number.

    Algorithm:
        - convert float to string
        - find position of decimal point relative to the length of the string

    Args:
        number (int/float)

    Returns:
        precission (int)

    Example:
        get_precission(5.00000)
        >> 1
        get_precission(5.12300)
        >> 3
    """
    if type(number) is str:
        number = number.rstrip("0")
    s = str(number)
    precission = s[::-1].find('.')
    return precission


def get_base(str_):
    """
    Get base (i.e. everything except extension) of string "str_".

    Args:
        str_ (str)

    Returns:
        base (str)

    Example:
        get_base("random_plot.png")
        >> "random_plot"
    """
    str_ = str(str_).rsplit("/", 1)[-1]
    base = str_.rsplit(".", 1)[0]
    return base


def get_extension(str_):
    """
    Get ".ext" (extension) of string "str_".

    Args:
        str_ (str)

    Returns:
        ext (str)

    Example:
        get_extension("random_plot.png")
        >> ".png"
    """
    str_ = str(str_).rsplit("/", 1)[-1]
    ext = "."+str_.rsplit(".", 1)[-1]
    return ext


def autodetect_header(fin):
    """
    Args:
        fin (str): input file (path)
    Returns:
        header_rows (int)
    """
    header_rows = 0
    extension = get_extension(fin)
    CHAR_SET = set("\n\t .,0123456789")

    with open(fin, "r") as f:
        for line in f:
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
    return header_rows


def read_file(fin, sep=None, usecols=(0, 1), skiprows='auto', dtype=np.float_):
    """
    Read file and return tuple of np.arrays with data for specified columns.

    Args:
        fin (str): input file (path)
        sep (None/str): seperator of columns.
            None: whitespace
        usecols (int or sequence): data columns
        skiprows ('auto'/int): ignore header rows of fin
            'auto' or -1: auto detect
        dtype (np.dtype cls): data type of returned np.array
            np.int_, np.int32, np.int64
            np.float_, np.float32, np.float64
            np.complex_, np.complex64, np.comples128

    Returns:
        if usecols==int:
            <np.array>: data of one column specified in usecols
        if usecols==sequence:
            list(<np.array>,<np.array>,...): data of each column specified in usecols

    Example:
        # single columns per fin
        X = read_file('data.txt', usecols=0)
        Y = read_file('data.txt', usecols=1)
        # multiple columns per fin
        X,Y = read_file('data.txt', usecols=(0,1))
    """
    if skiprows == "auto" or skiprows == -1:
        skiprows = autodetect_header(fin)

    if type(usecols) == int:
        col = np.loadtxt(fin, delimiter=sep, skiprows=skiprows, usecols=usecols, dtype=dtype)
        return col
    else:
        DATA = []
        for i in range(len(usecols)):
            col = np.loadtxt(fin, delimiter=sep, skiprows=skiprows, usecols=usecols[i], dtype=dtype)
            DATA.append(col)
        return DATA


def read_DCA_file(DCA_fin, n_DCA, usecols=(2, 3), skiprows='auto', filter_DCA=True):
    """
    Read DCA file and return DCA pairs IJ.

    Args:
        DCA_fin (str): DCA input file (path)
        n_DCA (int): number of used DCA contacts
        usecols (tuple/list): columns containing the RES pairs in DCA_fin
        skiprows ('auto'/int): ignore header rows of DCA_fin
            'auto' or -1: auto detect
        filter_DCA (bool):
            True: ignore DCA pairs with |i-j| < 4
            False: use all DCA pairs w/o applying filter

    Returns:
        DCA_PAIRS (list): DCA pairs IJ
    """
    I, J = read_file(DCA_fin, usecols=usecols, skiprows=skiprows, dtype=np.int_)
    DCA_PAIRS = []

    for k in range(len(I)):
        if len(DCA_PAIRS) == n_DCA:
            break
        if filter_DCA:
            if abs(I[k] - J[k]) < 4:
                pass    # ignore sites with indexdiff < 4
            else:
                DCA_PAIRS.append((I[k], J[k]))
        else:
            DCA_PAIRS.append((I[k], J[k]))

    return DCA_PAIRS


def get_PDBid(ref):
    """
    Obtains PDB id from ref by splitting its path and searching for alpha numerical (alnum) strings with length 4.

    Args:
        ref (str): reference path
        ref (MDA universe/atomgrp): reference structure

    Returns:
        if ref path contains one unique alnum string with length 4:
            return PDBid (str)
        if ref path contains multiple unique alnum strings with length 4:
            return PDBid (list)
        else:
            return
    """
    if type(ref) is str:
        ref = ref.replace('.', '/')
        ref = ref.replace('_', '/')
        split = ref.split('/')

    else:
        try:
            ref = ref.universe.filename
            ref = ref.replace('.', '/')
            ref = ref.replace('_', '/')
            split = ref.split('/')

        except AttributeError:
            print("get_PDBid(ref): AttributeError\
                 \nCan't obtain PDBid since ref is neither a path nor a MDA universe/atomgrp with the attribute ref.universe.filename")
            return

    PDBid = []   # list for guessed PDBids
    split = [i for i in split if i not in ["logs", "used", "from", "with"]]   # filter guessed PDBids

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
        PDBid = input_python_x('Enter correct PDB id:')
        return PDBid
    else:
        print("get_PDBid(ref): Warning\
             \nNo PDB id was found by splitting the ref path.")
        return


def print_table(data=[], prec=3, verbose=True, verbose_stop=30):
    """
    Prints each item of "data" elementwise next to each other as a table.

    Args:
        data (list of lists): table content, where each list corresponds to one column
        prec(None/int):
            (None): rounding off
            (int):  rounding on
        verbose (bool): print table
        verbose_stop (None/int): stop printing after N lines

    Returns:
        table_str (str): table formatted string. Can be used to save log files.

    Example:
        letters = ["A", "B", "C"]
        numbers = [1, 2, 3]
        table = misc.print_table([letters, numbers])
        >>  A	1
            B	2
            C	3

        # save table as log file:
        with open("table.log", "w") as fout:
            fout.write(table)
    """
    # apply precission to input
    if prec is not None:
        data = round_object(data, prec)

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
    if verbose:
        for item in table_str.splitlines()[:verbose_stop]:
            print(item)
        if (len(table_str.splitlines()) >= verbose_stop):
            print(f"misc.print_table(): printed only {verbose_stop} entries (set by verbose_stop parameter).")
    return table_str


def save_table(filename="", data=[], header="", default_dir="./logs", prec=3, **kwargs):
    """
    Executes misc.print_table() and saves the returned table string as a log file.

    Args:
        filename (str): filename or realpath
        data (list of lists): table content, where each list corresponds to one column
        header (str): header for log file, e.g. column descriptions
        default_dir (str)
        prec (None/int):
            (None): rounding off
            (int):  rounding on

    Kwargs:
        save_as (str): alias to filename

    Return:
        realpath (str): path to log file
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

        table_str = print_table(data, prec, verbose=False)
        fout.write(table_str)

    return realpath


def get_slice_indices(list1, list2):
    """
    Get slice indices of two lists (mainlist and sublist) where sublist is a slice of mainlist, e.g:
    sublist = mainlist[slice_ndx1:slice_ndx2]

    Args:
        list1 (list/array): mainlist or sublist
        list2 (list/array): mainlist or sublist

    Returns:
        slice_ndx1 (int): first slice index ("start")
        slice_ndx2 (int): second slice index ("end")
    """
    if len(list1) > len(list2):
        mainlist = list1
        sublist = list2
    else:
        mainlist = list2
        sublist = list1

    slice_ndx1 = 0
    for i in range(len(sublist)):
        if sublist[i] != mainlist[i]:
            mainlist = np.append(mainlist[1:], mainlist[:1])
            slice_ndx1 += 1
        else:
            break
    slice_ndx2 = slice_ndx1 + len(sublist)
    return (slice_ndx1, slice_ndx2)


def get_cutoff_array(array, cut_min=None, cut_max=None):
    """
    Applies cutoff on array and returns it as well as the corresponding indices.

    Args:
        array (array/list)
        cut_min (None/int/float): min value for cutoff
        cut_max (None/int/float): max value for cutoff

    Returns:
        cutoff_array (array/list)
        cutoff_array_ndx (array/list)

    Example:
        array = list(range(10,20))
        cut_min, cut_max = 12, 16
        cutoff_array, cutoff_array_ndx = get_cutoff_array(array, cut_min, cut_max)
        print(cutoff_array)
        >> [12, 13, 14, 15, 16]
        print(cutoff_array_ndx)
        >> [2, 3, 4, 5, 6]
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


def get_sub_array(array, ndx_sel):
    """
    Returns sub array of <array> for items with indices <ndx_sel>.

    Args:
        array (array/list): array-like object
        ndx_sel (array/list): index selection

    Returns:
        sub_array (array): sub array of array
    """
    sub_array = []
    for ndx in ndx_sel:
        sub_array.append(array[ndx])
    return np.array(sub_array)


def get_sorted_array(array):
    """
    Returns sorted array (increasing order) and the coresponding element indices of the input array.

    Args:
        array (array/list): array-like object

    Returns:
        SORTED_VALUES (array)
        SORTED_NDX (array)

    Example:
        A = np.array([1, 3, 2, 4])
        get_sorted_array(A)
        >> (array([1, 2, 3, 4]), array([0, 2, 1, 3]))
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    SORTED_ARRAY = np.sort(array)
    SORTED_NDX = np.argsort(array)

    return (SORTED_ARRAY, SORTED_NDX)


def get_ranked_array(array, reverse=False):
    """
    Returns ranked array (decreasing order) and the corresponding element indices of the input array.

    Args:
        array (array/list): array-like object
        reverse (bool):
            True: ranked array in increasing order (low to high)
            False: ranked array in decreasing order (high to low)

    Returns:
        RANKED_VALUES (array)
        RANKED_NDX (array)

    Example:
        A = np.array([1, 3, 2, 4])
        get_ranked_array(A)
        >> > (array([4, 3, 2, 1]), array([3, 1, 2, 0]))
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if reverse:
        RANKED_ARRAY = np.sort(array)
        RANKED_NDX = np.argsort(array)
    else:
        RANKED_ARRAY = np.flip(np.sort(array))
        RANKED_NDX = np.flip(np.argsort(array))

    return (RANKED_ARRAY, RANKED_NDX)


def get_quantile(data, p, prec=2, **kwargs):
    """
    Alias function of np.quantile()
    Get value (or score) below which p% of the observations may be found.

    Args:
        data (array-like)
        p (int/float): percent value ~ range 0-1
        prec (None/int): rounding precission

    Returns:
        quantile (int/float)
    """
    quantile = round(np.quantile(data, p, **kwargs), prec)
    return quantile


def get_percentile(data, p, prec=2, **kwargs):
    """
    Alias function of np.percentile()
    Get value (or score) below which p% of the observations may be found.

    Args:
        data (array-like)
        p (int/float): percent value ~ range 0-100
        prec (None/int): rounding precission

    Returns:
        percentile (int/float)
    """
    percentile = round(np.percentile(data, p, **kwargs), prec)
    return percentile
################################################################################
################################################################################
### custom plt.figure()


def figure(num=None, figsize=(6.5, 4), dpi=None, grid=[1, 1], hr=[], wr=[],
           palette=None, n_colors=None,
           context='notebook', style='ticks', font_scale=0,
           despine=True, offset=0, **kwargs):
    """
    Top level container for all the plot elements.
    Modified version of plt.figure() with gridspec and seaborn elements.
    "Example" below shows how to plot data by using misc.figure() and plt.sca() # set current axis

    Args:
        num (int): figure.number (to reuse the figure canvas)
        figsize (tuple)
        dpi (None/int): dpi settings of figure
            if monitor dpi is 100 and figure dpi is 300, then figure will be displayed
            on the monitor 3x larger (compared to printed version with e.g. 300 dpi printer).
        grid (list): nrows, ncols ~ height x width
        hr (list): height ratio of rows
        wr (list) width ratio of columns
        font_scale (float)
        palette (None/str/list/tuple):
            (None): use current palette
            (str): name of the palette, see seaborn documentation
            (list): list of colors
            (tuple): tuple of colors
        n_colors (None/int):
            (None): use default number of color cycles based on palette
            (int): remap number of color cycles for the palette
        context (None, dict, str): Affects sizes and widths of the plot, e.g. labelsize,
            linesize, etc. See return values of sns.plotting_context() to get all valid dict settings.
            (str):
                'notebook': scaled by 1.0
                'paper': scaled by 0.8
                'talk': scaled by 1.3
                'poster': scaled by 1.6
        style (None, dict, str): Affects aesthetic style of the plots, e.g. axes color,
            grid display, etc. See return values of sns.axes_style() to get all valid dict settings.
            (str):
                'ticks', 'dark', 'white', 'darkgrid', 'whitegrid'
        despine (bool): Remove the top and right spines from all axes. See sns.despine() documentation
            for individual axis spine removal.
        offset (dict/int): Absolute distance of spines. Use a dict to offset individual spines,
            e.g. offset={"bottom":0, "left":20}.

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)

    Example:
        fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        plt.sca(ax[0])
        plt.plot([0,1,2], [0,4,4])
        plt.plot([0,1,2], [0,10,10])
        plt.sca(ax[2])
        plt.plot([0,-1,-2], [0,-4,-4])
        plt.plot([0,-1,-2], [0,-10,-10])
    """
    if grid[0] == 1:
        hr = [1]
    if grid[1] == 1:
        wr = [1]

    if grid != [1, 1] and hr == []:
        hr = [1] * grid[0]
    if grid != [1, 1] and wr == []:
        wr = [1] * grid[1]

    if font_scale == 0:
        sns.set(context=context, style=style)  # context scales font elements by default
    else:
        sns.set(context=context, style=style, font_scale=font_scale)  # double scaling with font_scale
    sns.set_palette(palette, n_colors)

    if num is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    else:
        fig = plt.figure(num=num, figsize=figsize)

    gs = gridspec.GridSpec(grid[0], grid[1], width_ratios=wr, height_ratios=hr)
    ax = []

    for gs_item in gs:
        ax.append(plt.subplot(gs_item))

    if despine == True:
        sns.despine(offset=offset)

    plt.tight_layout()
    #plt.show()

    if len(ax) == 1:
        ax = ax[0]
    return (fig, ax)


def legend(labels=[""], handlecolors=[""], handlelength=1, handletextpad=None, loc=None, **kwargs):
    """
    Alias function for plt.legend() with most used parameter.

    Args:
        labels (sequence of strings)
        handles (sequence of `.Artist`)
        handlelength (None/int/float)
        handletextpad (None/int/float)
        loc (str):

            Location String   Location Code
            -------------------------------
            'best'            0
            'upper right'     1
            'upper left'      2
            'lower left'      3
            'lower right'     4
            'right'           5
            'center left'     6
            'center right'    7
            'lower center'    8
            'upper center'    9
            'center'          10

    Kwargs:
        # see help(plt.legend)
        title (str)
        edgecolor (str)
        fancybox (bool):
            True:  legendbox with round edges
            False: legendbox with normal edges
    """
    legend = plt.legend(labels=labels,
                        handlelength=handlelength,
                        handletextpad=handletextpad,
                        loc=loc, **kwargs)
    if handlecolors != [""]:
        for ndx, color in enumerate(handlecolors):
            legend.legendHandles[ndx].set_color(color)
    return


def savefig(filename, filedir="", create_dir=True, dpi=300):
    """
    general helpfunction:
        - save current figure
        - print message "Saved figure as: ..."

    Args:
        filename (None/str):
            (None): do not save figure
            (str): file name or realpath to file
        filedir (str): file directory
        create_dir (bool): create directory if it does not exist yet
        dpi (int): dpi settings

    Returns:
        realpath (str): realpath to saved figure
    """
    if filename == "" or filename is None:
        return
    if filedir != "":
        realpath = joinpath(filedir, filename, create_dir=create_dir)
    else:
        realpath = joinpath("./", filename, create_dir=create_dir)
    plt.savefig(realpath, dpi=dpi)
    print("Saved figure as:", realpath)
    return realpath


def autoapply_limits(fig_or_ax, margin=0.05):
    """
    Apply plt.xlim() and plt.ylim() to each axis object based on its xmin, xmax, ymin, ymax values.

    NOTES:
        - use only after plotting data
        - currently works only with Line2D data

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        margin (float): extra margin on upper limits, where margin=0.01 equals 1% of xmax and ymax

    Returns:
        if fig_or_ax is axis:
            xlim (list)
            ylim (list)
        if fig_or_ax is figure:
            XLIM (list of lists)
            YLIM (list of lists)

    Example:
        fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        plt.sca(ax[0])
        plt.plot([0,1,2], [0,4,4])
        plt.plot([0,1,2], [0,10,10])
        autoapply_limits(fig, margin=0.05)
        >> ([[0, 2.1], [0, 0], [0, 0], [0, 0]],
            [[0,10.5], [0, 0], [0, 0], [0, 0]])
    """
    # Test if object is figure or axis
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        axes = fig_or_ax.get_axes()
    if "axes" in locals():
        pass
    else:
        axes = [fig_or_ax]

    # get and apply limits
    XLIM = []
    YLIM = []

    for ax in axes:
        xlim = [0, 0]
        ylim = [0, 0]
        xmin, xmax = 0, 0
        ymin, ymax = 0, 0

        if isinstance(ax, matplotlib.axes._subplots.Axes):
            lines = ax.get_lines()

            # if Line2D data does not exist
            if len(lines) == 0:
                XLIM.append(xlim)
                YLIM.append(ylim)

            # if Line2D data does exist
            else:
                for l in lines:
                    xdata = l.get_xdata()
                    ydata = l.get_ydata()
                    xmin, xmax = np.amin(xdata), np.amax(xdata)
                    ymin, ymax = np.amin(ydata), np.amax(ydata)

                    if xmin < xlim[0]:
                        xlim[0] = xmin
                    if xmax > xlim[1]:
                        xlim[1] = xmax
                    if ymin < ylim[0]:
                        ylim[0] = ymin
                    if ymax > ylim[1]:
                        ylim[1] = ymax

                # save limits for each axis object
                if margin != 0:
                    xlim[1] = xlim[1] + margin*xlim[1]
                    ylim[1] = ylim[1] + margin*ylim[1]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                XLIM.append(xlim)
                YLIM.append(ylim)

    if len(XLIM) == 1:
        return (xlim, ylim)
    else:
        return (XLIM, YLIM)

################################################################################
################################################################################
# pickle functions


def pickle_dump(obj, filename='', pickledir='./pickle', overwrite=True, **kwargs):
    """
    bug: can't dump figures which used the misc.add_cbar() or misc.add_cbar_ax() function
         because of the use of ax.inset_axes().

    Create pickle directory and pickle.dump object as "<pickledir>/<filename>.pickle"

    Special cases:
        filename contains relative path: ignores pickledir input
        filename contains absolute path: ignores pickledir input

    Reason: intention is to dump file under the path "filename"

    Args:
        obj (matplotlib.figure.Figure/<any object>)
        filename (str): filename, hardcoded to add ".pickle" extension.
                        Can be relative or absolute path including the filename.
        pickledir (str): default pickle directory
        overwrite (bool): overwrite pickle dumped file if it already exists

    Kwargs:
        save_as (str): alias to filename

    Returns:
        filepath (str): realpath of dumped .pickle file
    """
    if "save_as" in kwargs:
        filename = kwargs["save_as"]
    if filename == "":
        raise TypeError("misc.pickle_dump(): missing 1 required argument: 'filename' or 'save_as'")
    # convention: save files with '.pickle' ending
    if filename[-7:] != ".pickle":
        filename = filename+".pickle"

    filepath = joinpath(filedir=pickledir, filename=filename, create_dir=True)
    if overwrite:
        if os.path.exists(filepath):
            os.remove(filepath)
    pl.dump(obj, open(filepath, 'wb'))
    if isinstance(obj, matplotlib.figure.Figure):
        print(f"pickle.dumped figure as:", os.path.realpath(filepath))
    else:
        print(f"pickle.dumped object as:", os.path.realpath(filepath))
    return filepath


def _pickle_get_ax_type(fig_or_ax):
    """
    Get axis type by counting <matplotlib.lines.Line2D> and <matplotlib.patches.Rectangle> objects.
    Additionally checks the ratio of Rectangle height:width to detect if barplot was created using
    plt.bar() or plt.barh().

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:
        ax_type (str):
            "line plot": ax was created using plt.plot()
            "bar plot":  ax was created using plt.bar()
            "barh plot": ax was created using plt.barh()
    """
    artists = fig_or_ax.get_default_bbox_extra_artists()

    n_lines = 0
    n_rectangles = 0
    maxh = 0
    maxw = 0

    for item in artists:
        if isinstance(item, matplotlib.lines.Line2D):
            n_lines += 1
        if isinstance(item, matplotlib.patches.Rectangle):
            n_rectangles += 1

            h = item.get_height()
            w = item.get_width()
            if h > maxh:
                maxh = h
            if w > maxw:
                maxw = w

    if (n_rectangles > 10) and (n_rectangles > n_lines):
        if maxh > maxw:
            ax_type = "bar plot"
        if maxw > maxh:
            ax_type = "barh plot"
    else:
        ax_type = "line plot"
    return ax_type


def _pickle_get_ax_data(fig_or_ax):
    """
    Get specific axis data:
        - limits
        - scaling
        - ticks
        - ticklabels

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing "Returns" part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        fig = fig_or_ax
        ax = fig.axes[0]
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        fig = fig_or_ax.figure
        ax = fig_or_ax

    ### ticklabels are "complicated"
    xticklabels = []
    temp = list(ax.get_xticklabels())
    for item in temp:
        xticklabels.append(item.get_text())

    yticklabels = []
    temp = list(ax.get_yticklabels())
    for item in temp:
        yticklabels.append(item.get_text())

    ax_data = {"ax_type": _pickle_get_ax_type(fig_or_ax),
               "xlim": ax.get_xlim(),
               "ylim": ax.get_ylim(),
               "xscale": ax.get_xscale(),
               "yscale": ax.get_yscale(),
               "xticks": ax.get_xticks(),
               "yticks": ax.get_yticks(),
               "xticklabels": xticklabels,
               "yticklabels": yticklabels}
    return ax_data


__pickle_get_ax_data___append_doc__ = """
        ax_data (dict): dict containing specific axis data
            dict keys:
                ax_type: "line plot": ax was created using plt.plot()
                         "bar plot":  ax was created using plt.bar()
                         "barh plot": ax was created using plt.barh()

                xlim            ylim
                xscale          yscale
                xticks          yticks
                xticklabels     yticklabels
"""


def _pickle_get_line_data(fig_or_ax):
    """
    Get <matplotlib.lines.Line2D> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing "Returns" part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    artists = fig_or_ax.get_default_bbox_extra_artists()
    line_data = []

    for item in artists:
        if isinstance(item, matplotlib.lines.Line2D):
            data_dict = {}
            data_dict["__description__"] = "line data ~ plt.plot()"
            data_dict["alpha"] = item.get_alpha()
            data_dict["color"] = item.get_color()
            data_dict["label"] = item.get_label()
            data_dict["ls"] = item.get_linestyle()
            data_dict["lw"] = item.get_linewidth()
            data_dict["marker"] = item.get_marker()
            data_dict["mec"] = item.get_mec()
            data_dict["mfc"] = item.get_mfc()
            data_dict["ms"] = item.get_markersize()
            data_dict["xdata"] = item.get_xdata()
            data_dict["ydata"] = item.get_ydata()
            line_data.append(data_dict)
    return line_data


__pickle_get_line_data___append_doc__ = """
        line_data (list): list of dicts, which contain data about <matplotlib.lines.Line2D> objects.
            dict keys:
                __description__: "line data ~ plt.plot()"

                alpha   ls (linestyle)  mac (marker edgecolor)  xdata
                color   lw (linewidth)  mfc (marker facecolor)  ydata
                label   marker          ms (markersize)
"""


def _pickle_get_rectangle_data(fig_or_ax):
    """
    Get <matplotlib.patches.Rectangle> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing "Returns" part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    artists = fig_or_ax.get_default_bbox_extra_artists()
    rect_data = []

    for item in artists:
        if isinstance(item, matplotlib.patches.Rectangle):
            data_dict = {}
            data_dict["__description__"] = "rectangle data ~ plt.bar() or plt.barh()"
            data_dict["alpha"] = item.get_alpha()
            data_dict["bbox"] = item.get_bbox()
            data_dict["bbox_points"] = item.get_bbox().get_points()
            data_dict["ec"] = item.get_ec()
            data_dict["fc"] = item.get_fc()
            data_dict["fill"] = item.get_fill()
            data_dict["height"] = item.get_height()
            data_dict["label"] = item.get_label()
            data_dict["ls"] = item.get_ls()
            data_dict["lw"] = item.get_lw()
            data_dict["width"] = item.get_width()
            data_dict["xy"] = item.get_xy()

            # ignore figure canvas rectangle
            if np.all(data_dict["bbox_points"] == np.array([[0, 0], [1, 1]])):
                pass
            else:
                rect_data.append(data_dict)
    return rect_data


__pickle_get_rectangle_data___append_doc__ = """
        rect_data (list): list of dicts, which contain data about <matplotlib.patches.Rectangle> objects.
            dict keys:
                __description__: "rectangle data ~ plt.bar()"
                xy: userland coordinates of bottom left rectangle corner
                    (rectangle is defined by xy, width, height)

                alpha               fill                width
                bbox                height
                bbox_points         label
                ec (edgecolor)      ls (linestyle)
                fc (facecolor)      lw (linewidth)
"""
__pickle_get_rectangle_data___bugs_doc__ = """
    Current bugs/problems:
        EDGECOLOR:
        case 1) if plt.bar(..., ec='None', alpha=0.6) was used, then item.get_ec()
                returns array for black color but with alpha value of 0
        case 2) if figure is pickle.dumped and then loaded, the edgecolor values
                are displayed correct (i.e. black with alpha of 0)
        case 3) if plt.bar(...,ec=<read value> , alpha=0.6) is used, all edge
                colors get an alpha value of 0.6

        Although this is somehow expected, but case 1) and case 3) should still
        yield the same results.

        -> workaround is coded in misc.pickle_plot()
    """


def pickle_load(filename, pickledir="./pickle", plot=False):
    """
    pickle.load figure from "<pickledir>/<filename>.pickle". If the pickle file
    contains figure data, auto detects the figure type (i.e. if created using
    plt.plot(), plt.bar() or plt.barh()).

    Args:
        filename (str): realpath to .pickle file or filename within pickledir
        pickledir (str): default pickle directory
        plot (bool)

    Returns:
        if pickle file is a figure object:
            ax_data (dict)
            line_data (list)
            rect_data (list)

        else:
            object (?)

    Notes:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing "Returns" part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    # test if filename is "fullpath" or "relative path to pickle_dir"
    filepath = joinpath(filedir=pickledir, filename=filename, create_dir=False)
    obj = pl.load(open(filepath, "rb"))

    if isinstance(obj, matplotlib.figure.Figure):
        fig = obj
        ax_data = _pickle_get_ax_data(fig)
        line_data = _pickle_get_line_data(fig)
        rect_data = _pickle_get_rectangle_data(fig)
        if not plot:
            fig.set_size_inches(0, 0)
            plt.close()
        return (ax_data, line_data, rect_data)

    else:
        return obj


__pickle_load___example_doc__ = """
    Example:
        ax_data, line_data, rect_data = pickle_load( < ... > )
        # acces data
        line_data[0]["xdata"]
        line_data[0]["color"]
"""


def pickle_plot(pickle_files=[], import_settings=True, xscale='auto', yscale='auto',
                align_ylim=True, hist_minorticks=False, **kwargs):
    """
    Note: Code for "align" parameter is currently only intended for figures with
          grid = [<any>, 2], i.e. width of 2 figures.

    Args:
        pickle_files (str/list of str): path to pickle file(s) containing the (sub)figure data
        import_settings (bool): apply settings (for line and rectangle objects) from imported pickle_files
        xscale (str): 'auto', 'linear', 'log', 'symlog', 'logit'
            'auto': apply xscale of the used pickle_files
        yscale (str): 'auto', 'linear', 'log', 'symlog', 'logit'
            'auto': apply yscale of the used pickle_files
        align_ylim (bool): align y axis limits of figures on the same row
        hist_minorticks (bool): turns minorticks (~logscale) for hist figure on/off

    Kwargs:
        # see args of misc.figure()

    Returns:
        fig (matplotlib.figure.Figure)
        ax (ax/list of axes ~ matplotlib.axes._subplots.Axes)
    """
    default = {"grid": [1, 1]}
    cfg = CONFIG(default, **kwargs)
    if len(pickle_files) != 1 and "grid" not in kwargs:
        nrows = int(round_up(len(pickle_files)/2, base=1))
        ncols = 2
        cfg.grid = [nrows, ncols]

    if isinstance(pickle_files, type("")):
        pickle_files = [pickle_files]

    if pickle_files == [] or not isinstance(pickle_files, type([])):
        exec("help(pickle_plot)")
        raise TypeError("Invalid type of 'pickle_files'. Read the docstring above.")

    fig, ax = figure(**cfg)
    for ndx, pickle in enumerate(pickle_files):
        # skip pickle_files with empty path string
        if pickle == "":
            continue  # python: continue with NEXT loop iteration
        else:
            if len(pickle_files) == 1:
                plt.sca(ax)
            else:
                plt.sca(ax[ndx])
            pass  # python: continue with CURRENT loop iteration aka "do nothing"

        ax_data, line_data, rect_data = pickle_load(pickle)

        ### reconstruct figure
        if ax_data["ax_type"] == "line plot":
            for item in line_data:
                if import_settings:
                    plt.plot(item["xdata"], item["ydata"],
                             alpha=item["alpha"], color=item["color"],
                             label=item["label"], ls=item["ls"], lw=item["lw"],
                             marker=item["marker"], ms=item["ms"],
                             mec=item["mec"], mfc=item["mfc"])
                else:
                    plt.plot(item["xdata"], item["ydata"])

        if ax_data["ax_type"] == "bar plot" or ax_data["ax_type"] == "barh plot":
            for item in rect_data:
                if import_settings:
                    ### workaround start: weird bug with edge colors not properly importing the alpha value
                    if item["ec"] == (0, 0, 0, 0):
                        temp_ec = "None"
                    else:
                        temp_ec = item["ec"]
                    ### workaround end

                    if ax_data["ax_type"] == "bar plot":
                        plt.bar(item["xy"][0], height=item["height"], width=item["width"],
                                fc=item["fc"], ec=temp_ec, alpha=item["alpha"],
                                fill=item["fill"], ls=item["ls"], lw=item["lw"])
                    elif ax_data["ax_type"] == "barh plot":
                        plt.barh(item["xy"][1], height=item["height"], width=item["width"],
                                 fc=item["fc"], ec=temp_ec, alpha=item["alpha"],
                                 fill=item["fill"], ls=item["ls"], lw=item["lw"])

                if not import_settings:
                    if ax_data["ax_type"] == "bar plot":
                        plt.bar(item["xy"][0], height=item["height"], width=item["width"])
                    elif ax_data["ax_type"] == "barh plot":
                        plt.barh(item["xy"][1], height=item["height"], width=item["width"])

        ### general settings
        plt.xlim(ax_data["xlim"])
        plt.ylim(ax_data["ylim"])
        if xscale == "auto":
            plt.xscale(ax_data["xscale"])
        else:
            plt.xscale(xscale)
        if yscale == "auto":
            plt.yscale(ax_data["yscale"])
        else:
            plt.yscale(yscale)

    if align_ylim:
        for ndx in range(cfg.grid[0]*cfg.grid[1]):
            if ndx % cfg.grid[1] == 0:
                ref_ax = ax[ndx]
            else:
                target_ax = ax[ndx]
            if ndx > 0:
                align_limits(ref_ax=ref_ax, target_ax=target_ax, apply_on='y')
        """
        if "grid" in kwargs:
            for ndx in range(kwargs["grid"][0]*kwargs["grid"][1]):
                i = ndx//kwargs["grid"][1]
                j = i + ndx % kwargs["grid"][1]
                if i != j:
                    align_limits(ref_ax=ax[i], target_ax=ax[j], apply_on='y')
        else:
            for i in range(0, len(ax), 2):
                align_limits(ref_ax=ax[i], target_ax=ax[i+1], apply_on='y')
        """
    if not hist_minorticks:
        plt.minorticks_off()

    plt.tight_layout()
    plt.show()

    #fig = plt.gcf()
    #ax = fig.axes
    return (fig, ax)


def hide_figure(fig_or_ax=None, num=None, close=True):
    """
    "Hide" figure by setting its size to (0, 0) inches.
    If number is passed, apply

    Note: hide_figure() and hide_plot() are aliases.

    Args:
        fig_or_ax (None/matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        num (None/int): if integer is passed, hide figure with this number
        close (bool): if True, close figure instead of just hiding
    """
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        fig = fig_or_ax
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        fig = fig_or_ax.get_figure()
    elif fig_or_ax is None and num is not None:
        fig = plt.figure(num)

    fig.set_size_inches(0, 0)
    if close and num is None:
        plt.close(fig.number)
    if close and num is not None:
        plt.close(num)

    return


# alias function of hide_figure()
hide_plot = hide_figure


#################################################################################
################################################################################
### limits, ticks and labels functions for plot aesthetics of gridplots

def align_limits(ref_ax, target_ax, apply_on='y', new_lim=[]):
    """
    - Read limits of ref_ax and asign them to target_ax
    - if "new_lim" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y'
        new_lim (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_lim == []:
            ax.set_xlim(ax_data["xlim"])
        else:
            ax.set_xlim(new_lim)

    def _apply_on_y(ax):
        if new_lim == []:
            ax.set_ylim(ax_data["ylim"])
        else:
            ax.set_ylim(new_lim)
    ##################################################
    if apply_on == "y":
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    elif apply_on == "x":
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)

    return


def align_ticks(ref_ax, target_ax, apply_on='y', new_ticks=[]):
    """
    - Read ticks of ref_ax and asign them to target_ax
    - if "new_ticks" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y'
        new_ticks (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_ticks == []:
            ax.set_xticks(ax_data["xticks"])
        else:
            ax.set_xticks(new_ticks)

    def _apply_on_y(ax):
        if new_ticks == []:
            ax.set_yticks(ax_data["yticks"])
        else:
            ax.set_yticks(new_ticks)
    ##################################################
    if apply_on == "y":
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    elif apply_on == "x":
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)

    return


def align_ticklabels(ref_ax, target_ax, apply_on='y', new_ticklabels=[]):
    """
    - Read tickalbels of ref_ax and asign them to target_ax
    - if "new_ticklabels" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y'
        new_ticklabels (list): assign new values to both ref_ax and target_ax
    """
    ax_data = _pickle_get_ax_data(ref_ax)
    ##################################################
    # nested functions

    def _apply_on_x(ax):
        if new_ticklabels == []:
            ax.set_xticklabels(ax_data["xticklabels"])
        else:
            ax.set_xticklabels(new_ticklabels)

    def _apply_on_y(ax):
        if new_ticklabels == []:
            ax.set_yticklabels(ax_data["yticklabels"])
        else:
            ax.set_yticklabels(new_ticklabels)
    ##################################################
    if apply_on == "y":
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    elif apply_on == "x":
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    return


def apply_shared_axes(ax, grid):
    """
    - removes xticklabels of all axes except for bottom row.
    - removes yticklabels of all axes except for left column.

    Args:
        ax (list of axes ~ matplotlib.axes._subplots.Axes)
        grid (list): nrows, ncols ~ height x width
    """
    ndx = list(range(grid[0]*grid[1]))

    # grid indices of left column
    ndx_left = ndx[::grid[1]]
    # grid indices of bottom row
    ndx_bottom = ndx[grid[0]*grid[1]-grid[1]:]

    for item in [ax[i] for i in ndx if i not in set(ndx_left)]:
        item.set_yticklabels([])
    for item in [ax[i] for i in ndx if i not in set(ndx_bottom)]:
        item.set_xticklabels([])
    return


class number_base_factorization():
    """
    Class to get 10base factorization of any number,
    e.g. 123 = 3*10^0 + 2*10^1 + 1*10^2

    Example:
        x = number_base_factorization(123)
        x() # show content of x
        >> self.number: 123
           self.sign: +
           self.digits: [1, 2, 3]
           self.pos_base10_factors: [3, 2, 1]
           self.neg_base10_factors: None
    """

    def __init__(self, num):
        def get_sign(num):
            """
            signum function
            """
            if num >= 0:
                return "+"
            else:
                return "-"

        def get_digits(num):
            """
            Convert <num> into list of digits.
            """
            digits = [int(d) if d not in set("-.") else d for d in str(num)]
            if "-" in digits:
                del digits[digits.index("-")]
            return digits

        def base10_factorize(digits):
            """
            get base10 factorization of a list with digits.

            Args:
                digits (list)

            Returns:
                pos_base10_factors (list)
                neg_base10_factors (list)
            """
            if "." in digits:
                temp = digits.index(".")
                pos_base10_factors = digits[:temp][::-1]
                neg_base10_factors = digits[temp+1:]
            else:
                pos_base10_factors = digits[::-1]
                neg_base10_factors = None
            return pos_base10_factors, neg_base10_factors

        self.number = num
        self.sign = get_sign(num)
        self.digits = get_digits(num)
        temp = base10_factorize(self.digits)
        self.pos_base10_factors = temp[0]
        self.neg_base10_factors = temp[1]

    def __call__(self):
        """
        Print class content.
        """
        print("self.number:", self.number)
        print("self.sign:", self.sign)
        print("self.digits:", self.digits)
        print("self.pos_base10_factors:", self.pos_base10_factors)
        print("self.neg_base10_factors:", self.neg_base10_factors)
        return


def setup_ticks(vmin, vmax, major_base, minor_base=None):
    """
    Setup axis ticks of a plot in vmin <= ticks <= vmax.

    Args:
        vmin (int/float)
        vmax (int/float)
        major_base (int/float)
        minor_base (int/float)

    Returns:
        majorticks (list)
        minorticks (list)
    """
    min_ = round_down(vmin, base=major_base)
    max_ = round_up(vmax, base=major_base)

    majorticks = np.arange(min_, max_+major_base, major_base)
    majorticks = [i for i in majorticks if i >= vmin if i <= vmax]
    if minor_base is not None:
        minorticks = np.arange(min_+minor_base, max_, minor_base)
        minorticks = [i for i in minorticks if i not in majorticks]
    else:
        minorticks = list()
    return majorticks, minorticks


def setup_logscale_ticks(vmax):
    """
    Setup majorticks and minorticks for logscale with ticks <= vmax.

    Args:
        vmax (int/float)

    Returns:
        majorticks (list): ticks at 1, 10, 100, 1000, ....
        minorticks (list): ticks at 2, 3, 4,..., 8, 9,
                                   20,30,40,...,80,90,
                                   ...
    """
    vmax_factorized = number_base_factorization(vmax)
    majorticks = [[1]]
    minorticks = []

    for ndx, item in enumerate(vmax_factorized.pos_base10_factors):
        major, minor = setup_ticks(vmin=10**ndx, vmax=10**(ndx+1), major_base=10**(ndx+1), minor_base=10**ndx)
        majorticks.append(major)
        minorticks.append(minor)

    majorticks = flatten_array(majorticks)
    minorticks = flatten_array(minorticks)
    return majorticks, minorticks


def set_logscale_ticks(ax, axis="x", vmax=None, minorticks=True, **kwargs):
    """
    Apply logscale ticks on specific axis of ax.
    If vmax is specified, sets vmax as upper limit of axis.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        axis (str): "x", "y", "xy"
        vmin (None/int/float): highest logscale tick value
        minorticks (bool): display minorticks on/off

    Kwargs:
        xmin (None/int/float)
        xmax (None/int/float)
        ymin (None/int/float)
        ymax (None/int/float)
    """
    default = {"vmax": vmax,
               "xmin": None,
               "xmax": None,
               "ymin": None,
               "ymax": None}
    cfg = CONFIG(default, **kwargs)

    if "x" in axis:
        if cfg.vmax is None:
            _, vmax = ax.get_xlim()
        major, minor = setup_logscale_ticks(vmax)
        ax.set_xscale("log")
        if cfg.xmin is None:
            cfg.xmin, _ = ax.get_xlim()
        if cfg.xmax is None:
            _, cfg.xmax = ax.get_xlim()
        ax.set_xticks([], minor=True)
        ax.set_xticks(major)
        if minorticks:
            ax.set_xticks(minor, minor=True)

        # set upper limit via vmax (not cfg.vmax)
        if vmax is None:
            ax.set_xlim(cfg.xmin, cfg.xmax)
        else:
            ax.set_xlim(cfg.xmin, vmax)

    if "y" in axis:
        if cfg.vmax is None:
            _, vmax = ax.get_ylim()
        major, minor = setup_logscale_ticks(vmax)
        ax.set_yscale("log")
        if cfg.ymin is None:
            cfg.ymin, _ = ax.get_ylim()
        if cfg.ymax is None:
            _, cfg.ymax = ax.get_ylim()
        ax.set_yticks([], minor=True)
        ax.set_yticks(major)
        if minorticks:
            ax.set_yticks(minor, minor=True)

        # set upper limit via vmax (not cfg.vmax)
        if vmax is None:
            ax.set_ylim(cfg.ymin, cfg.ymax)
        else:
            ax.set_ylim(cfg.ymin, vmax)

    return

#################################################################################
################################################################################
### colorbar functions


def add_cbar_ax(ax, bounds="auto", location="right", orientation="vertical"):
    """
    Add colorbar ax relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        bounds (str/list):
            (str): "auto": apply bounds based on kws 'orientation' and 'location'.
            (list): [x0, y0, height, width] using axes coordinates.
                    Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Returns:
        cbar_ax (matplotlib.axes._axes.Axes): colorbar ax
    """
    # apply passed bounds and ignore kws 'orientation' and 'location'
    if bounds != "auto":
        pass

    # apply bounds based on kws 'orientation' and 'location'
    else:
        if location == "right":
            x0, y0 = 1.05, 0
        elif location == "left":
            x0, y0 = -0.1, 0
        elif location == "top":
            x0, y0 = 0, 1.1
        elif location == "bottom":
            x0, y0 = 0, -0.2

        if orientation == "horizontal":
            h, w = 0.05, 1
        elif orientation == "vertical":
            h, w = 1, 0.05

        bounds = [x0, y0, w, h]

    cbar_ax = ax.inset_axes(bounds)
    return cbar_ax


def add_cbar(ax, cbar_ax=None, bounds="auto", location="right", orientation="vertical", **kwargs):
    """
    Draw colorbar relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        cbar_ax (None/matplotlib.axes._axes.Axes): colorbar ax
            (None): add new cbar_ax relative to existing ax
            (matplotlib.axes._axes.Axes): use passed cbar_ax (can be created using misc.add_cbar_ax()).
        bounds (str/list):
            (str): "auto": apply bounds based on kws 'orientation' and 'location'.
            (list): [x0, y0, height, width] using axes coordinates.
                    Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Kwargs:
        cbar_label/label (str)
        cbar_fontweight/fontweight (str): "normal", "bold"
        cbar_location/location (str): "right", "bottom", "left", "top"
        cbar_orientation/orientation (str): "vertical", "horizontal"

    Returns:
        cbar (matplotlib.colorbar.Colorbar)
    """
    default = {"cbar_label": "",
               "cbar_fontweight": "normal",
               "cbar_location": location,
               "cbar_orientation": orientation
               }

    cfg = CONFIG(default, **kwargs)
    cfg.update_by_alias(alias="label", key="cbar_label", **kwargs)
    cfg.update_by_alias(alias="fontweight", key="cbar_fontweight", **kwargs)

    if cbar_ax is None:
        cbar_ax = add_cbar_ax(ax, bounds,
                              location=cfg.cbar_location,
                              orientation=cfg.cbar_orientation)

    for item in ax.get_children():
        if isinstance(item, matplotlib.collections.QuadMesh):
            QuadMesh = item  # QuadMesh ~ sns.heatmap() plots

    if "QuadMesh" in locals():
        # mappable must be matplotlib.cm.ScalarMappable ~ sns.heatmap()
        # see help(fig.colorbar) for more info
        cbar = plt.colorbar(mappable=QuadMesh, cax=cbar_ax, orientation=cfg.cbar_orientation)
    else:
        #print("misc.add_cbar() works currently only with sns.heatmap() plots.")
        cbar = plt.colorbar(cax=cbar_ax, orientation=cfg.cbar_orientation)

    cbar.set_label(label=cfg.cbar_label, weight=cfg.cbar_fontweight)
    cbar_set_ticks_position(cbar, location=cfg.cbar_location)
    return cbar


def cbar_set_ticks_position(cbar, location):
    """
    Set ticks position of colorbar.

    Args:
        cbar (matplotlib.colorbar.Colorbar)
        location (str): "right", "bottom", "left", "top"
    """
    if location == "top" or location == "bottom":
        cbar.ax.xaxis.set_ticks_position(location)
    if location == "right" or location == "left":
        cbar.ax.yaxis.set_ticks_position(location)
    return
#################################################################################
################################################################################
### complete / append docstrings


def INIT_append_docstrings():
    """
    Some functions in this module have split up docstrings.
    The intention of this design choice is to assure that docstrings which rely
    on each other are matching and always up to date. This function is executed
    ONCE upon initialization of the module and appends the missing parts.
    """
    _pickle_get_ax_data.__doc__ += __pickle_get_ax_data___append_doc__
    _pickle_get_line_data.__doc__ += __pickle_get_line_data___append_doc__
    _pickle_get_rectangle_data.__doc__ += __pickle_get_rectangle_data___append_doc__ + __pickle_get_rectangle_data___bugs_doc__

    for item in [__pickle_get_ax_data___append_doc__,
                 __pickle_get_line_data___append_doc__,
                 __pickle_get_rectangle_data___append_doc__,
                 __pickle_load___example_doc__]:
        pickle_load.__doc__ += item
    return


INIT_append_docstrings()
