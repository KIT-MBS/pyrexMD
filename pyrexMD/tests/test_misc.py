# @Author: Arthur Voronin <arthur>
# @Date:   10.05.2021
# @Filename: test_misc.py
# @Last modified by:   arthur
# @Last modified time: 21.06.2021


import pyrexMD.misc as misc
import MDAnalysis as mda
import numpy as np
from numpy.testing import assert_allclose
import pytest
import os

if os.path.exists("./files"):
    pre = "."
else:
    pre = "./tests/"


def test_round_object():
    A = [0.1234567890, 1.9876543210]
    val = misc.round_object(A, prec=3)
    expected = [0.123, 1.988]
    assert assert_allclose(val, expected) == None

    B = np.array([0.1234567890, 1.9876543210])
    val = misc.round_object(B, prec=3)
    expected = np.array([0.123, 1.988])
    assert assert_allclose(val, expected) == None
    return


def test_get_substrings():
    val = misc.get_substrings("split/this_string.into:parts")
    expected = ['split', 'this', 'string', 'into', 'parts']
    assert val == expected

    val = misc.get_substrings("split/this_string.into:parts", reverse=True)
    expected = ['parts', 'into', 'string', 'this', 'split']
    assert val == expected
    return


def test_split_lists():
    A = list(range(10))
    val = misc.split_lists(A, 2)
    expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 2, 4, 6, 8], [1, 3, 5, 7, 9])
    assert val == expected

    A = list(range(10))
    val = misc.split_lists(A, 2, remap=True)
    expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert val == expected

    A = list(range(10))
    val = misc.split_lists(A, 4)
    expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6, 8, 9], [3, 7])
    assert val == expected

    A = list(range(10))
    val = misc.split_lists(A, 4, remap=True)
    expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1])
    assert val == expected
    return


def test_get_precision():
    assert misc.get_precision(5.00000) == 1
    assert misc.get_precision(5.12300) == 3
    return


def test_get_base():
    assert misc.get_base("random_plot.png") == "random_plot"
    assert misc.get_base("/bla/blubb/random_plot.png") == "random_plot"
    return


def test_get_extension():
    assert misc.get_extension("random_plot.png") == ".png"
    assert misc.get_extension("/bla/blubb/random_plot.png") == ".png"
    return


def test_insert_str():
    str1 = "str1 is this"
    str2 = "str2 bla"
    val = misc.insert_str(str1, str2, sep=" ", loc="after")
    expected = "str1 str2 blais this"
    assert val == expected

    val = misc.insert_str(str1, str2, sep=" ", loc="before")
    expected = "str1str2 bla is this"
    assert val == expected
    return


def test_autodetect_header():
    assert misc.autodetect_header(f"{pre}/files/header_file1.txt") == 8
    assert misc.autodetect_header(f"{pre}/files/header_file2.txt") == 10
    return


def test_read_file():
    val = misc.read_file(f"{pre}/files/header_file1.txt")
    expected = [np.array([1., 2., 3., 4., 5.]), np.array([1., 2., 3., 4., 5.])]
    assert assert_allclose(val, expected) == None

    val = misc.read_file(f"{pre}/files/header_file2.txt")
    expected = [np.array([1., 2., 3., 4., 5.]), np.array([10., 20., 30., 40., 50.])]
    assert assert_allclose(val, expected) == None
    return


def test_read_DCA_ile():
    val = misc.read_DCA_file(f"{pre}/files/2hda.score", n_DCA=50)
    expected = np.load(f"{pre}/files/READ_DCA_FILE.npy", allow_pickle=True)
    assert assert_allclose(val[0], expected[0]) == None
    assert assert_allclose(val[1], expected[1]) == None
    return


def test_get_PDBid():
    ref = mda.Universe(f"{pre}/files/1l2y/1l2y_ref.pdb")
    val = misc.get_PDBid(ref)
    assert val == "1l2y"
    return


def test_get_slice_indices():
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_slice_indices(A, [1, 2]) == (1, 3)
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_slice_indices(A, [3, 9, 8]) == (3, 6)
    A = [0, 1, 2, 3, 9, 8, 7]
    with pytest.raises(misc.Error):
        assert misc.get_slice_indices(A, [3, 8, 8])  # no slice possible
    return


def test_get_subarray_start_ndx():
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_subarray_start_ndx(A, [1, 2]) == 1
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_subarray_start_ndx(A, [3, 9, 8]) == 3
    return


def test_get_subarray():
    A = [0, 1, 2, 3, 9, 8, 7]
    assert (misc.get_subarray(A, [0, 1, 2]) == [0, 1, 2]).all()
    A = [0, 1, 2, 3, 9, 8, 7]
    assert (misc.get_subarray(A, [5, 6]) == [8, 7]).all()
    A = [0, 1, 2, 3, 9, 8, 7]
    assert (misc.get_subarray(A, [0, 2, 4]) == [0, 2, 9]).all()
    A = [0, 1, 2, 3, 9, 8, 7]
    assert (misc.get_subarray(A, [3, 2, 1]) == [3, 2, 1]).all()
    return


def test_get_sorted_array():
    A = np.array([1, 3, 2, 4])
    val = misc.get_sorted_array(A)
    assert (val[0] == [1, 2, 3, 4]).all()
    assert (val[1] == [0, 2, 1, 3]).all()
    return


def test_get_ranked_array():
    A = np.array([1, 3, 2, 4])
    val = misc.get_ranked_array(A, reverse=False, verbose=False)
    assert (val[0] == [4, 3, 2, 1]).all()
    assert (val[1] == [3, 1, 2, 0]).all()
    A = np.array([1, 3, 2, 4])
    val = misc.get_ranked_array(A, reverse=True, verbose=False)
    assert (val[0] == [1, 2, 3, 4]).all()
    assert (val[1] == [0, 2, 1, 3]).all()
    return


def test_CONFIG():
    default = {"start": None,
               "stop": 100,
               "step": 1,
               "color": "red"}
    change = {"color": "blue",
              "marker": "."}
    target = {"start": None,
              "stop": 100,
              "step": 1,
              "color": "blue",
              "marker": "."}
    cfg1 = misc.CONFIG(default, **change)
    cfg2 = misc.CONFIG(target)
    assert cfg1.items() == cfg2.items()
