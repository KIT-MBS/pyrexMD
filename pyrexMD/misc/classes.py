# @Author: Arthur Voronin <arthur>
# @Date:   26.08.2021
# @Filename: class.py
# @Last modified by:   arthur
# @Last modified time: 26.08.2021

"""
.. hint:: This module is a collection of general classes which are used
    frequently to streamline pyrexMD.
"""


import os
import sys
import time
import copy


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
            raise Warning("dtypeError class usage: <arg> must be str")
        if not isinstance(dtype, str):
            raise Warning("dtypeError class usage: <dtype> must be str")

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
