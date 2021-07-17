# @Author: Arthur Voronin
# @Date:   17.04.2021
# @Filename: misc.py
# @Last modified by:   arthur
# @Last modified time: 19.07.2021

"""
This module is a collection of miscellaneous functions.
"""

# miscellaneous
import PIL
import pdf2image
import pickle as pl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import shutil
import glob
import sys
import subprocess
import time
import copy
import termcolor


def apply_matplotlib_rc_settings():
    """
    apply matplotlib rc settings:

      - render math text including greek letters in italic font by default
      - render math text including greek letters in non-italic font using \\mathrm{}
      - does not require an external LaTeX installation

    Code:

      - matplotlib.rcParams['font.size'] = 12
      - matplotlib.rcParams['font.weight'] = "normal"
      - matplotlib.rcParams['font.family'] = 'sans-serif'
      - matplotlib.rcParams['font.sans-serif'] = 'Arial'
      - matplotlib.rcParams['mathtext.fontset'] = 'custom'
      - matplotlib.rcParams['mathtext.rm'] = 'sans'
      - matplotlib.rcParams['mathtext.it'] = 'sans:italic'
      - matplotlib.rcParams['mathtext.default'] = 'it'
    """
    matplotlib.rcParams['font.size'] = 12
    matplotlib.rcParams['font.weight'] = "normal"
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'sans'
    matplotlib.rcParams['mathtext.it'] = 'sans:italic'
    matplotlib.rcParams['mathtext.default'] = 'it'
    return


def update_alias_docstring(f1, f2):
    """
    Use an existing docstring as template and replace 'Alias function of <alias>' part

      - determine docstring template based on length
      - test if docstring template contains "Alias function of <alias>"
      - copy and replace alias such that:
      | f1 docstring contains: Alias function of <f2.__name__>
      | f2 docstring contains: Alias function of <f1.__name__>

    Args:
        f1 (function): function with docstring (template)
        f2 (function): function with no/short docstring (replace with updated template)
    """
    if not hasattr(f1, '__call__') or not hasattr(f2, '__call__'):
        raise Error("f1 and f2 must be functions.")
        return
    if f1.__doc__ is None:
        f1, f2 = f2, f1
    if f1.__doc__ is None:
        raise Error("f1 and f2 are both without docstring.")
    try:
        if len(f1.__doc__) < len(f2.__doc__):
            f1, f2 = f2, f1
    except TypeError:
        pass

    f2.__doc__ = f1.__doc__.replace(f"Alias function of {f2.__name__}", f"Alias function of {f1.__name__}")
    if f"Alias function of {f2.__name__}" not in f1.__doc__:
        raise Warning(f"{f1.__name__} docstring does not contain 'Alias function of {f2.__name__}'.")
    if f"Alias function of {f1.__name__}" not in f2.__doc__:
        raise Warning(f"{f2.__name__} docstring does not contain 'Alias function of {f1.__name__}'.")
    return


################################################################################
################################################################################
### CUSTOM CLASSES

class Error(Exception):
    """Base class for exceptions"""
    pass


class dtypeError(Exception):
    """
    Base class for dtype exceptions

    Example:
        | >> misc.dtypeError("source", "str")
        | TypeError: <source> must be str

    """

    def __init__(self, arg, dtype):
        if not isinstance(arg, str):
            raise Warning(f"dtypeError class usage: <arg> must be str")
        if not isinstance(dtype, str):
            raise Warning(f"dtypeError class usage: <dtype> must be str")

        raise TypeError(f"<{arg}> must be {dtype}")
    pass


class HiddenPrints:
    """
    Class to hide print commands.

    Args:
        verbose (bool):

          | True: show prints
          | False: hide prints (default)

    Example:
        | with HiddenPrints():
        |   print("This print is hidden")
        | with HiddenPrints(verbose=True):
        |   print("This print is shown")
        | This print is shown
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        return


class HiddenPrints_ALL:
    """
    Class to hide all stdout and stderr prints.

    Args:
        verbose (bool):
          | True: show prints
          | False: hide prints (default)

    Example:
        | with HiddenPrints_ALL():
        |   print("This print is hidden")
        | with HiddenPrints(verbose=True):
        |   print("This print is shown")
        | This print is shown
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

        return


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
        Alias of print_config() method.
        """
        self.print_config()
        return

    def __setitem__(self, key, value):
        """
        Set config object key via [] ~ backets, e.g. cfg["key"] = value
        """
        self.__setattr__(key, value)
        return

    def __delitem__(self, key):
        """
        Delete config object key via [] ~ backets, e.g. del cfg["key"]
        """
        self.__delattr__(key)
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

    def deepcopy_without(self, keys):
        """
        Return deep copy of cfg object but without passed keys.

        Args:
            keys (str, list, tuple):
              | str: single key
              | list, tuple: multiple keys
        """
        obj = copy.deepcopy(self)
        # case 1: keys is a single key ~str
        if isinstance(keys, str):
            try:
                obj.pop(keys)
            except KeyError:
                pass
        # case 2: keys is a list of keys
        if isinstance(keys, (list, tuple)):
            for key in keys:
                try:
                    obj.pop(key)
                except KeyError:
                    pass
        return obj

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
          | default = {"colors": ["g", "r"],
          |            "ms": 1}
          | alias_dict = {"color_positive": "teal",
          |               "color_negative": "orange",
          |               "markersize": 5}
          | cfg = misc.CONFIG(default, **alias_dict)
          | cfg.print_config()
          | >>  key                 value
          |     colors              ['g', 'r']
          |     ms                  1
          |     color_positive      teal
          |     color_negative      orange
          |     markersize          5
          |
          | cfg.update_by_alias(alias="markersize", key="ms", **alias_dict)
          | cfg.update_by_alias(alias="color_positive", key="colors", key_ndx=0, **alias_dict)
          | cfg.update_by_alias(alias="color_negative", key="colors", key_ndx=1, **alias_dict)
          | cfg.print_config()
          | >>  key                 value
          |     colors              ['teal', 'orange']
          |     ms                  5
          |     color_positive      teal
          |     color_negative      orange
          |     markersize          5
        """
        if alias in kwargs:
            if key_ndx is None:
                self.__dict__[key] = kwargs[alias]
            else:
                self.__dict__[key][key_ndx] = kwargs[alias]
        return

    def pop(self, key):
        """
        remove key from dict and return its value
        """
        return self.__dict__.pop(key)

    def print_config(self):
        """
        Print and return stored data in table format with "key" and "value" columns.
        """
        _str = f"{'key':20}{'value':20}\n\n"

        for item in self.__dict__.items():
            _str += f"{item[0]:20}{item[1]}\n"
        print(_str)

        return _str


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


def timeit(timer=None, msg="elapsed time:"):
    """
    Test elapsed time of a process

    Args:
        timer (None, TIMER CLASS)
        msg (str): message

    Returns:
        timer (TIMER CLASS)

    Example:

      | t = TIMER()
      | timeit(t)
      | <code>
      | timeit(t)
    """
    if timer == None:
        timer = TIMER()
        timer.t0 = timer._get_time()
    elif timer.t0 == 0:
        timer.t0 = timer._get_time()
    elif timer.t0 != 0:
        timer.t1 = timer._get_time()

    if timer.t1 != 0:
        t_diff = round(timer.t1 - timer.t0, 3)
        print(f"{msg} {t_diff} s")

    return timer


################################################################################
################################################################################
### linux-like cmds ~ os package

def cwd(verbose=True):
    """
    Alias function of pwd(). Get/Print current working directory.

    Args:
        verbose (bool): print cwd

    Returns:
        cwd (str)
            current working directory
    """
    if verbose:
        cprint(f"cwd: {os.getcwd()}", "blue")
    return os.getcwd()


def pwd(verbose=True):
    return cwd(verbose=verbose)


update_alias_docstring(cwd, pwd)


def isdir(path):
    """
    Return true if the path refers to an existing directory.
    """
    return os.path.isdir(path)


def isfile(path):
    """
    Return true if the path refers to an existing file.
    """
    return os.path.isfile(path)


def pathexists(path):
    """
    Return true if path exists.
    """
    return os.path.exists(path)


def realpath(path):
    """
    Return realpath.
    """
    return os.path.realpath(path)


def relpath(path):
    """
    Return relative path.
    """
    return os.path.relpath(path)


def dirpath(path, realpath=True):
    """
    Alias function of get_filedir(). Get realpath or relpath of the last directory of <path>.

    Args:
        path (str)

    Returns:
        dirpath (str)
    """
    if realpath:
        return os.path.dirname(os.path.realpath(path))
    else:
        return os.path.dirname(os.path.relpath(path))


### Alias function of dirpath
def get_filedir(path, realpath=True):
    return dirpath(path, realpath=realpath)


update_alias_docstring(dirpath, get_filedir)


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

    real_path = realpath(path)
    dir_path = dirpath(path)
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
    target = realpath(target)

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


### Alias function of rm()
def remove(path, pattern=None, verbose=True):
    return rm(path, pattern, verbose)


update_alias_docstring(rm, remove)


def bash_cmd(cmd, verbose=False):
    """
    Execute any bash command via python.

    .. Note:: works with pipe.

    Args:
        cmd (str): bash cmd
        verbose (bool): print executed shell command

    Example:
        | >> bash_cmd("ls")
        | >> bash_cmd("touch test")
        | >> bash_cmd("ls")
        | test
    """
    if verbose:
        print("executing shell command:", cmd)

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)

    # wait until subprocess is finished
    while True:
        p.poll()
        if p.returncode is not None:
            break
    return p.returncode


################################################################################
################################################################################
### pillow/PIL stuff


def convert_image(fin, fout, format="auto", **kwargs):
    """
    Converts image type. Currently ONLY tested for

        - pdf -> png
        - png -> tiff
        - tiff -> png


    Args:
        fin (str): file path for file in (including extension)
        fout (str): file path for file out (including extension)
        format (str):
          | fout format/extension
          | "auto"
          | "png"

    Keyword Args:
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
    convert multiple images. Currently ONLY tested for

        - pdf -> png
        - png -> tiff
        - tiff -> png

    Args:
        folder_in (str):  file path for folder with input  images
        folder_out (str): file path for folder with output images
        format (str):
          | fout format/extension
          | "auto"
          | "png"

    Keyword Args:
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

def cprint(msg, color=None, on_color=None, attr=None, **kwargs):
    """
    Alias function of termcolor.cprint(). Apply colored print.

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


def UNZIP(iterable):
    """
    alias function of zip(\*iterable)

    Args:
        iterable

    Returns:
        zip(\*iterable)
    """
    return zip(*iterable)


def unzip(iterable):
    """
    Unzips iterable into two objects.
    """
    A = list(UNZIP(iterable))[0]
    B = list(UNZIP(iterable))[1]
    return (A, B)


def get_python_version():
    """
    Returns python version.

    Returns:
        version (str):
            python version in format major.minor.micro (e.g. 3.8.5)

    """
    major = sys.version_info[0]
    minor = sys.version_info[1]
    micro = sys.version_info[2]
    version = f"{major}.{minor}.{micro}"
    return version


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
            table_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data[0][ndx], data[1][ndx],
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


def get_sorted_array(array):
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

    Example:
        | >> A = np.array([1, 3, 2, 4])
        | >> get_sorted_array(A)
        | (array([1, 2, 3, 4]), array([0, 2, 1, 3]))
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    SORTED_ARRAY = np.sort(array)
    SORTED_NDX = np.argsort(array)

    return (SORTED_ARRAY, SORTED_NDX)


def get_ranked_array(array, reverse=False, verbose=True, **kwargs):
    """
    Returns ranked array (decreasing order) and the corresponding element indices of the input array.

    Args:
        array (array, list): array-like object
        reverse (bool):
          | True:  ascending ranking order (low to high)
          | False: decending ranking order (high to low)
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
################################################################################
################################################################################
### custom plt.figure()


def figure(num=None, figsize=(6.5, 4), dpi=None, grid=[1, 1], hr=[], wr=[],
           palette=None, n_colors=None,
           context='notebook', style='ticks', font_scale=0,
           despine=True, offset=0, **kwargs):
    """
    Top level container for all the plot elements. Modified version of plt.figure()
    with gridspec and seaborn elements. "Example" below shows how to plot data
    by using misc.figure() and plt.sca() # set current axis

    Args:
        num (int): figure.number (to reuse the figure canvas)
        figsize (tuple)
        dpi (None, int):
          | dpi settings of figure
          | if monitor dpi is 100 and figure dpi is 300, then figure will be displayed
          | on the monitor 3x larger (compared to printed version with e.g. 300 dpi printer).
        grid (list): nrows, ncols ~ height x width
        hr (list): height ratio of rows
        wr (list) width ratio of columns
        font_scale (float)
        palette (None, str, list, tuple):
          | None: use current palette
          | str: name of the palette, see seaborn documentation
          | list: list of colors
          | tuple: tuple of colors
        n_colors (None, int):
          | None: use default number of color cycles based on palette
          | int: remap number of color cycles for the palette
        context (None, dict, str):
          | Affects sizes and widths of the plot, e.g. labelsize,
          | linesize, etc. See return values of sns.plotting_context() to get
          | all valid dict settings.
          | str:'notebook': scaled by 1.0
          |     'paper': scaled by 0.8
          |     'talk': scaled by 1.3
          |     'poster': scaled by 1.6
        style (None, dict, str):
          | Affects aesthetic style of the plots, e.g. axes color,
          | grid display, etc. See return values of sns.axes_style() to get all
          | valid dict settings.
          | str:'ticks', 'dark', 'white', 'darkgrid', 'whitegrid'
        despine (bool):
          | Remove the top and right spines from all axes. See sns.despine()
          | documentation for individual axis spine removal.
        offset (dict/int):
          | Absolute distance of spines. Use a dict to offset individual spines,
          | e.g. offset={"bottom":0, "left":20}.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes

    Example:
        | fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        | plt.sca(ax[0])
        | plt.plot([0,1,2], [0,4,4])
        | plt.plot([0,1,2], [0,10,10])
        | plt.sca(ax[2])
        | plt.plot([0,-1,-2], [0,-4,-4])
        | plt.plot([0,-1,-2], [0,-10,-10])
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


def scatter(x, y, z=None, **kwargs):
    """
    Creates scatter plot. Applies cmap for z values if z is passed.

    Args:
        x (array)
        y (array)
        z (None, array)

    Keyword Args:
        figsize (tuple)
        aspect ('auto', 'equal', 'int'):

        marker (None, str)
        ms (None, int): marker size
        cmap (str):
          | colormap name, e.g. 'virids', 'plasma', 'inferno', 'summer', 'winter', 'cool', etc.
          | You can reverse the cmap by appending '_r' to the name.
          | See https://matplotlib.org/stable/tutorials/colors/colormaps.html
        vmin (None, float): min value of cmap and colorbar
        vmax (None, float): max value of cmap and colorbar
        cbar_label (None, str)
        xlabel (None, str)
        ylabel (None, str)

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"figsize": (6.6, 6.6),
               "aspect": "auto",
               "marker": None,
               "ms": None,
               "cmap": "viridis",
               "vmin": None,
               "vmax": None,
               "cbar_label": None,
               "xlabel": None,
               "ylabel": None}
    cfg = CONFIG(default, **kwargs)
    ###########################################################
    fig, ax = figure(**cfg)
    plt.scatter(x=x, y=y, c=z, s=cfg.ms, marker=cfg.marker, cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax)
    plt.xlabel(cfg.xlabel, fontweight="bold")
    plt.ylabel(cfg.ylabel, fontweight="bold")
    add_cbar(ax=ax, **cfg)
    ax.set_aspect(cfg.aspect)
    plt.tight_layout()
    return (fig, ax)


def set_pad(fig_or_ax, xpad=None, ypad=None):
    """
    Set pad (spacing) between axis and axis labels

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        xpad (None, float):
          | spacing between xaxis and its labels
          | None: use current settings
        ypad (None, float):
          | spacing between yaxis and its labels
          | None: use current settings
    """
    default = {"xpad": 3.5,
               "ypad": 3.5}
    cfg = CONFIG(default, xpad=xpad, ypad=ypad)
    ############################################################################
    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        if len(fig_or_ax.axes) == 1:
            ax = [fig_or_ax.axes]
        else:
            ax = fig_or_ax.axes
    elif isinstance(fig_or_ax, matplotlib.axes._subplots.Axes):
        ax = [fig_or_ax]
    elif isinstance(fig_or_ax, list):
        ax = fig_or_ax

    for ax in ax:
        if xpad is not None:
            for tick in ax.get_xaxis().get_major_ticks():
                tick.set_pad(cfg.xpad)
        if ypad is not None:
            for tick in ax.get_yaxis().get_major_ticks():
                tick.set_pad(cfg.ypad)
    return


def legend(labels=[""], handlecolors=[""], handlelength=1, handletextpad=None, loc=None, **kwargs):
    """
    Alias function of plt.legend() with most frequently used parameters.

    Args:
        labels (sequence of strings)
        handlescolors (list)
        handlelength (None/int/float)
        handletextpad (None/int/float)
        loc (str):
          | Location String   Location Code
          | -------------------------------
          | 'best'            0
          | 'upper right'     1
          | 'upper left'      2
          | 'lower left'      3
          | 'lower right'     4
          | 'right'           5
          | 'center left'     6
          | 'center right'    7
          | 'lower center'    8
          | 'upper center'    9
          | 'center'          10

    Keyword Args:
        title (str)
        edgecolor (str)
        fancybox (bool):
          | True:  legendbox with round edges
          | False: legendbox with normal edges
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
    general helpfunction

      - save current figure
      - print message "Saved figure as: ..."

    Args:
        filename (None, str):
          | None: do not save figure
          | str: file name or realpath to file
        filedir (str): file directory
        create_dir (bool): create directory if it does not exist yet
        dpi (int): dpi settings

    Returns:
        realpath (str)
            realpath to saved figure
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


### alias function
save_fig = savefig


def autoapply_limits(fig_or_ax, margin=0.05):
    """
    Apply plt.xlim() and plt.ylim() on each axis object based on its xmin, xmax,
    ymin, ymax values.

    .. NOTE::

      - use only after plotting data
      - currently works only with Line2D data

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        margin (float): extra margin on upper limits, where margin=0.01 equals 1% of xmax and ymax

    Returns:
        xlim (list)
            if fig_or_ax is axis
        ylim (list)
            if fig_or_ax is axis
        XLIM (list of lists)
            if fig_or_ax is figure
        YLIM (list of lists)
            if fig_or_ax is figure

    Example:
        | >> fig, ax = misc.figure(grid=[2,2], hr=[1,1], wr=[2,1])
        | >> plt.sca(ax[0])
        | >> plt.plot([0,1,2], [0,4,4])
        | >> plt.plot([0,1,2], [0,10,10])
        | >> autoapply_limits(fig, margin=0.05)
        | ([[0, 2.1], [0, 0], [0, 0], [0, 0]],
        | [[0,10.5], [0, 0], [0, 0], [0, 0]])
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


def pickle_dump(obj, filename='', pickledir='./pickle',
                overwrite=True, verbose=True, **kwargs):
    """
    bug: can't dump figures which used the misc.add_cbar() or misc.add_cbar_ax() function
         because of the use of ax.inset_axes().

    Create pickle directory and pickle.dump object as "<pickledir>/<filename>.pickle"

    Special cases:

      - filename contains relative path: ignores pickledir input
      - filename contains absolute path: ignores pickledir input

    Reason:

        intention is to dump file under the path "filename"

    Args:
        obj (matplotlib.figure.Figure/<any object>)
        filename (str):
          | filename, hardcoded to add ".pickle" extension.
          | Can be relative or absolute path including the filename.
        pickledir (str): default pickle directory
        overwrite (bool): overwrite pickle dumped file if it already exists
        verbose (bool)

    Keyword Args:
        save_as (str): alias of filename

    Returns:
        filepath (str)
            realpath of dumped .pickle file
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
    with open(filepath, 'wb') as file:
        pl.dump(obj, file)
    if verbose:
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
        ax_type (str)
            | "line plot" ~ ax was created using plt.plot()
            | "bar plot"  ~ ax was created using plt.bar()
            | "barh plot" ~ ax was created using plt.barh()
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
    # NOTE 2: Missing doc string part is coded right after this function.
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
        ax_data (dict)
            dictionary containing specific axis data, see table below

            +-------------+--------------+-------------------------------------+
            | key         | value        | description                         |
            +-------------+--------------+-------------------------------------+
            |ax_type (str)| "line plot"  | ax was created using plt.plot()     |
            +             +--------------+-------------------------------------+
            |             | "bar plot"   | ax was created using plt.bar()      |
            +             +--------------+-------------------------------------+
            |             | "barh plot"  | ax was created using plt.barh()     |
            +-------------+--------------+-------------------------------------+
            |xlim/ylim    | tuple        | content of ax.get_xlim()            |
            +-------------+--------------+-------------------------------------+
            |xscale/yscale| str          | content of ax.get_xscale()          |
            +-------------+--------------+-------------------------------------+
            |xticks/yticks| array        | content of ax.get_xticks()          |
            +-------------+--------------+-------------------------------------+
            |x/yticklabels| list         | text content of ax.get_xticklabels()|
            +-------------+--------------+-------------------------------------+
"""


def _pickle_get_line_data(fig_or_ax):
    """
    Get <matplotlib.lines.Line2D> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
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
        line_data (list)
            list of dicts, which contain data about <matplotlib.lines.Line2D>
            objects with viable Keyword Args as shown in the table below

            +---------+----------------+-------------------------+--------+
            | alpha   | ls (linestyle) | mac (marker edgecolor)  | xdata  |
            +---------+----------------+-------------------------+--------+
            | color   | lw (linewidth) | mfc (marker facecolor)  | ydata  |
            +---------+----------------+-------------------------+--------+
            | label   | marker         | ms (markersize)         |        |
            +---------+----------------+-------------------------+--------+
"""


def _pickle_get_rectangle_data(fig_or_ax):
    """
    Get <matplotlib.patches.Rectangle> objects data of figure/axis.

    Args:
        fig_or_ax (matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)

    Returns:"""
    # NOTE 1: End this docstring with |Returns:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
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
        rect_data (list)
            list of dicts, which contain data about <matplotlib.patches.Rectangle>
            objects with viable Keyword Args as shown in the table below

            +----------------+--------+--------+
            | alpha          | fill   | width  |
            +----------------+--------+--------+
            | bbox           | height |  xy    |
            +----------------+--------+--------+
            | bbox_points    | label  |        |
            +----------------+--------+--------+
            | ec (edgecolor) | ls (linestyle)  |
            +----------------+-----------------+
            | fc (facecolor) | lw (linewidth)  |
            +----------------+-----------------+

        .. Hint:: xy are the userland coordinates starting from bottom left
           rectangle corner (rectangle is defined by xy, width, height)
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
    contains figure data, auto detects the figure type (i.e. it created using
    plt.plot(), plt.bar() or plt.barh()).

    .. Warning:: %matplotlib notebook backend leaves empty space below figure in
       jupyter notebook when closing figs which were loaded via pickle files.

    Args:
        filename (str): realpath to .pickle file or filename within pickledir
        pickledir (str): default pickle directory
        plot (bool)

    Returns:
        ax_data (dict)
            if pickle file is a figure object
        line_data (list)
            if pickle file is a figure object
        rect_data (list)
            if pickle file is a figure object
        object (?)
            else

    .. Note ::"""
    # NOTE 1: End this docstring with |Notes:"""| to remove trailing newline
    # NOTE 2: Missing doc string part is coded right after this function.
    #         It will be appended once here and to other docstrings during
    #         the first initialization of the module.
    #         -> see INIT_append_docstrings()

    # test if filename is "fullpath" or "relative path to pickle_dir"
    filepath = joinpath(filedir=pickledir, filename=filename, create_dir=False)
    with open(filepath, "rb") as file:
        obj = pl.load(file)
        if isinstance(obj, matplotlib.figure.Figure):
            fig = obj
            ax_data = _pickle_get_ax_data(fig)
            line_data = _pickle_get_line_data(fig)
            rect_data = _pickle_get_rectangle_data(fig)
            if plot == False:
                fig.set_size_inches(0, 0)
                plt.close()
            return (ax_data, line_data, rect_data)

        else:
            return obj


__pickle_load___append_doc__ = """
    Example:
        | ax_data, line_data, rect_data = pickle_load( < filename > )
        |  # acces data
        | line_data[0]["xdata"]
        | line_data[0]["color"]
"""


def pickle_plot(pickle_files=[], import_settings=True, xscale='auto', yscale='auto',
                align_ylim=True, hist_minorticks=False, **kwargs):
    """
    Creates multifigure from loading pickle_files

    .. Warning:: data is first loaded -> creates a figure -> closed. However in
       jupyter %matplotlib notebook backend leaves empty space below figure when
       closing figs which were loaded via pickle. The empty space is only visible
       in the notebook (i.e. saving a pickle_plot figure is working fine.)

    .. Note: Code for "align" parameter is currently only intended for figures with
        grid = [<any>, 2], i.e. width of 2 figures.

    Args:
        pickle_files (str,list of str): path to pickle file(s) containing the (sub)figure data
        import_settings (bool): apply settings (for line and rectangle objects) from imported pickle_files
        xscale (str):
          | 'auto', 'linear', 'log', 'symlog', 'logit'
          | 'auto': apply xscale of the used pickle_files
        yscale (str):
          | 'auto', 'linear', 'log', 'symlog', 'logit'
          | 'auto': apply yscale of the used pickle_files
        align_ylim (bool): align y axis limits of figures on the same row
        hist_minorticks (bool): turns minorticks (~logscale) for hist figure on/off

    .. Hint:: Args and Keyword Args of misc.figure() are valid.

    Returns:
        fig (class)
            matplotlib.figure.Figure
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
    """
    default = {"grid": [1, 1]}
    cfg = CONFIG(default, **kwargs)
    if len(pickle_files) != 1 and "grid" not in kwargs:
        nrows = int(round_up(len(pickle_files)/2, base=1))
        ncols = 2
        cfg.grid = [nrows, ncols]

    if isinstance(pickle_files, str):
        pickle_files = [pickle_files]

    elif pickle_files == [] or not isinstance(pickle_files, list):
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
        fig = plt.gcf()
        ax = fig.axes
        for ndx in range(cfg.grid[0]*cfg.grid[1]):
            if ndx % cfg.grid[1] == 0:
                ref_ax = ax[ndx]
            else:
                target_ax = ax[ndx]
            if ndx > 0:
                align_limits(ref_ax=ref_ax, target_ax=target_ax, apply_on='y')
        # """
        # if "grid" in kwargs:
        #     for ndx in range(kwargs["grid"][0]*kwargs["grid"][1]):
        #         i = ndx//kwargs["grid"][1]
        #         j = i + ndx % kwargs["grid"][1]
        #         if i != j:
        #             align_limits(ref_ax=ax[i], target_ax=ax[j], apply_on='y')
        # else:
        #     for i in range(0, len(ax), 2):
        #         align_limits(ref_ax=ax[i], target_ax=ax[i+1], apply_on='y')
        # """
    if not hist_minorticks:
        plt.minorticks_off()

    plt.tight_layout()
    plt.show()

    #fig = plt.gcf()
    #ax = fig.axes
    return (fig, ax)


def hide_figure(fig_or_ax=None, num=None, close=True):
    """
    Alias function of hide_plot(). "Hide" figure by setting its size to (0, 0)
    inches. If number is passed, apply on figure with that number.

    Args:
        fig_or_ax (None, matplotlib.figure.Figure/matplotlib.axes._subplots.Axes)
        num (None, int): if integer is passed, hide figure with this number
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


# alias function of hide_figure
def hide_plot(fig_or_ax=None, num=None, close=True):
    return(hide_figure(fig_or_ax=fig_or_ax, num=num, close=close))


update_alias_docstring(hide_plot, hide_figure)


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
        apply_on (str): 'x', 'y', 'xy'
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
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def align_ticks(ref_ax, target_ax, apply_on='y', new_ticks=[]):
    """
    - Read ticks of ref_ax and asign them to target_ax
    - if "new_ticks" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y', 'xy'
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
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def align_ticklabels(ref_ax, target_ax, apply_on='y', new_ticklabels=[]):
    """
    - Read ticklabels of ref_ax and asign them to target_ax
    - if "new_ticklabels" is passed: assign new values to both ref_ax and target_ax

    Args:
        ref_ax (matplotlib.axes._subplots.Axes)
        target_ax (matplotlib.axes._subplots.Axes)
        apply_on (str): 'x', 'y', 'xy'
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
    if "x" in apply_on:
        _apply_on_x(ref_ax)
        _apply_on_x(target_ax)
    if "y" in apply_on:
        _apply_on_y(ref_ax)
        _apply_on_y(target_ax)
    return


def apply_shared_axes(ax, grid):
    """
    - removes xticklabels of all axes except for bottom row.
    - removes yticklabels of all axes except for left column.

    Args:
        ax (class, list)
            ax or list of axes ~ matplotlib.axes._subplots.Axes
        grid (list)
            grid of figure (example: grid=[2,2] for 2x2 figure or grid =[3,2] for 3x2 figure)
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


def convert_ticklabels(axes, multiplier, apply_on='y', prec=0):
    """
    Read ticklabels of axes and multiply with "multiplier"

    Args:
        axes (list of matplotlib.axes._subplots.Axes)
        multiplier (int/float): multiplier for conversion, ie: new_tickvalue = multiplier * old_tickvalue
        apply_on (str): 'x', 'y', 'xy'
        prec (int):
          | precision
          | 0: use integers with no decimal precision
          | n: float with n decimal precision
    """
    for ax in axes:
        if "x" in apply_on:
            # read
            xticklabels = []
            temp = list(ax.get_xticklabels())
            for item in temp:
                xticklabels.append(item.get_text())
            # convert
            if prec == 0:
                xticklabels = [int(float(item)*multiplier) for item in xticklabels if item != ""]
            elif prec > 0:
                xticklabels = [round(float(item)*multiplier, prec) for item in xticklabels if item != ""]
            ax.set_xticklabels(xticklabels)

        if "y" in apply_on:
            # read
            yticklabels = []
            temp = list(ax.get_yticklabels())
            for item in temp:
                yticklabels.append(item.get_text())

            # convert
            if prec == 0:
                yticklabels = [int(float(item)*multiplier) for item in yticklabels if item != ""]
            elif prec > 0:
                yticklabels = [round(float(item)*multiplier, prec) for item in yticklabels if item != ""]
            ax.set_yticklabels(yticklabels)
    return


class number_base_factorization():
    """
    Class to get 10base factorization of any number,
    e.g. 123 = 3*10^0 + 2*10^1 + 1*10^2

    Example:
      | >> x = number_base_factorization(123)
      | >> x() # show content of x
      | self.number: 123
      | self.sign: +
      | self.digits: [1, 2, 3]
      | self.pos_base10_factors: [3, 2, 1]
      | self.neg_base10_factors: None
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
                    list with positive factors
                neg_base10_factors (list)
                    list with negative factors
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
        vmin (int, float)
        vmax (int, float)
        major_base (int, float)
        minor_base (int, float)

    Returns:
        majorticks (list)
            list with majorticks
        minorticks (list)
            list with minorticks
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
        vmax (int, float)

    Returns:
        majorticks (list)
            ticks at 1, 10, 100, 1000, etc.
        minorticks (list):
            ticks at 2, 3, 4,..., 8, 9, 20,30,40,...,80,90, etc.
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


def set_logscale_ticks(ax, apply_on="x", vmax=None, minorticks=True, **kwargs):
    """
    Apply logscale ticks on ax. If vmax is specified, sets vmax as upper limit.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        apply_on (str): "x", "y", "xy"
        vmin (None, int, float): highest logscale tick value
        minorticks (bool): display minorticks on/off

    Keyword Args:
        xmin (None, int, float)
        xmax (None, int, float)
        ymin (None, int, float)
        ymax (None, int, float)
    """
    default = {"vmax": vmax,
               "xmin": None,
               "xmax": None,
               "ymin": None,
               "ymax": None}
    cfg = CONFIG(default, **kwargs)

    if "x" in apply_on:
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

    if "y" in apply_on:
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


def create_cmap(seq, vmin=None, vmax=None, ax=None):
    """
    Return a LinearSegmentedColormap

    Args:
        seq (sequence): sequence of color strings and floats. The floats describe the color thresholds and
           should be increasing and in the interval (0,1).
        vmin (float): min value of cmap
        vmax (float): max value of cmap
        ax (None, ax): figure ax for adding colorbar

    Returns:
        cmap (LinearSegmentedColormap)

    Example:
        seq = ["lightblue", 2/6, "lightgreen", 3/6, "yellow", 4/6, "orange", 5/6, "red"]
        cmap = misc.create_cmap(seq, vmin=0, vmax=12)
    """
    for ndx, item in enumerate(seq):
        if isinstance(item, str):
            seq[ndx] = matplotlib.colors.ColorConverter().to_rgb(item)
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    cmap = matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)

    # add colorbar if ax is passed
    if ax is not None:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = 1
        _norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=_norm, cmap=cmap), ax=ax)
    return cmap


def add_cbar_ax(ax, bounds="auto", location="right", orientation="vertical"):
    """
    Add colorbar ax relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        bounds (str,list):
          | str: "auto": apply bounds based on kws 'orientation' and 'location'.
          | list: [x0, y0, height, width] using axes coordinates.
          |       Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Returns:
        cbar_ax (matplotlib.axes._axes.Axes)
            colorbar ax
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


def add_cbar(ax, cbar_ax=None, cmap=None, bounds="auto", location="right", orientation="vertical", **kwargs):
    """
    Draw colorbar relative to existing ax.

    Args:
        ax (matplotlib.axes._subplots.Axes)
        cbar_ax (None/matplotlib.axes._axes.Axes):
          | colorbar ax
          | None: add new cbar_ax relative to existing ax
          | matplotlib.axes._axes.Axes: use passed cbar_ax (can be created using misc.add_cbar_ax()).
        cmap (None, LinearSegmentedColormap): output of create_cmap()
        bounds (str, list):
          | str: "auto": apply bounds based on kws 'orientation' and 'location'.
          | list: [x0, y0, height, width] using axes coordinates.
          |       Ignores kws 'orientation' and 'location'.
        location (str): "right", "bottom", "left", "top"
        orientation (str): "vertical", "horizontal"

    Keyword Args:
        vmin (None, float): colorbar min value
        vmax (None, float): colorbar max value
        cbar_label/label (None, str)
        cbar_fontweight/fontweight (None, str): "bold", "normal"
        cbar_location/location (None, str): "right", "bottom", "left", "top"
        cbar_orientation/orientation (None, str): "vertical", "horizontal"

    Returns:
        cbar (matplotlib.colorbar.Colorbar)
            color bar
    """
    default = {"vmin": None,
               "vmax": None,
               "cbar_label": None,
               "cbar_fontweight": "bold",
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
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        if cfg.vmin is None:
            cfg.vmin = 0
        if cfg.vmax is None:
            cfg.vmax = 1
        _norm = matplotlib.colors.Normalize(vmin=cfg.vmin, vmax=cfg.vmax)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=_norm, cmap=cmap), cax=cbar_ax, orientation=cfg.cbar_orientation)
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
                 __pickle_load___append_doc__]:
        pickle_load.__doc__ += item
    return


INIT_append_docstrings()
