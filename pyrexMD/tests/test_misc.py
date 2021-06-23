# @Author: Arthur Voronin <arthur>
# @Date:   10.05.2021
# @Filename: test_misc.py
# @Last modified by:   arthur
# @Last modified time: 23.06.2021


import pyrexMD.misc as misc
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from unittest.mock import patch
import pytest
import os


cwd = misc.cwd(verbose=False)

# cwd is <...>/pyrexMD/tests/
if misc.realpath(f"{cwd}").split("/")[-1] == "tests":
    pre = "."
    pre2 = "./files/figs/"
    pre3 = "../examples/files/protein/"
    pre4 = "../examples/files/pickle/"

# cwd is <...>/pyrexMD/
elif misc.realpath(f"{cwd}").split("/")[-1] == "pyrexMD":
    pre = "./tests/"
    pre2 = "./tests/files/figs/"
    pre3 = "./examples/files/protein/"
    pre4 = "./examples/files/pickle/"


################################################################################
################################################################################
### important functions


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

    # coverage
    misc.split_lists(10, 2)
    misc.split_lists(range(10), 2)
    with pytest.raises(TypeError) as e_info:
        misc.split_lists("wrong_dtype", 2)
    return


def test_get_precision():
    assert misc.get_precision(5.00000) == 1
    assert misc.get_precision("5.12300") == 3
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

    # coverage
    with pytest.raises(misc.Error) as e_info:
        misc.insert_str(str1, str2, sep=" ", loc="")
    return


def test_read_file():
    val = misc.read_file(f"{pre}/files/header_file1.txt")
    expected = [np.array([1., 2., 3., 4., 5.]), np.array([1., 2., 3., 4., 5.])]
    assert assert_allclose(val, expected) == None

    val = misc.read_file(f"{pre}/files/header_file2.txt")
    expected = [np.array([1., 2., 3., 4., 5.]), np.array([10., 20., 30., 40., 50.])]
    assert assert_allclose(val, expected) == None

    # coverage
    misc.read_file(f"{pre}/files/header_file1.txt", skiprows=None, dtype=list)
    misc.read_file(f"{pre}/files/2hba/2hba.score", usecols=(0, 1), dtype=[int, float])
    return


def test_read_DCA_ile():
    val = misc.read_DCA_file(f"{pre}/files/2hba/2hba.score", n_DCA=50)
    expected = np.load(f"{pre}/files/READ_DCA_FILE.npy", allow_pickle=True)
    assert assert_allclose(val[0], expected[0]) == None
    assert assert_allclose(val[1], expected[1]) == None
    return


def test_get_PDBid():
    ref = mda.Universe(f"{pre}/files/1l2y/1l2y_ref.pdb")
    val = misc.get_PDBid(ref)
    assert val == "1l2y"

    # coverage
    misc.get_PDBid("wrong_dtype")
    misc.get_PDBid("no_pdbid_in_string")
    return


def test_get_slice_indices():
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_slice_indices(A, [1, 2]) == (1, 3)
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_slice_indices(A, [3, 9, 8]) == (3, 6)
    A = [0, 1, 2, 3, 9, 8, 7]
    with pytest.raises(misc.Error):
        assert misc.get_slice_indices(A, [3, 8, 8])  # no slice possible

    # coverage
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_slice_indices([1, 2], A) == (1, 3)
    return


def test_get_cutoff_array():
    array = list(range(10, 20))
    cut_min, cut_max = 12, 16
    cutoff_array, cutoff_array_ndx = misc.get_cutoff_array(array, cut_min=cut_min, cut_max=cut_max)
    assert (cutoff_array == [12, 13, 14, 15, 16])
    assert (cutoff_array_ndx == [2, 3, 4, 5, 6])

    # coverage
    cutoff_array, cutoff_array_ndx = misc.get_cutoff_array(array, cut_max=cut_max)
    assert (cutoff_array == [10, 11, 12, 13, 14, 15, 16])
    assert (cutoff_array_ndx == [0, 1, 2, 3, 4, 5, 6])
    return


def test_get_subarray_start_ndx():
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_subarray_start_ndx(A, [1, 2]) == 1
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_subarray_start_ndx(A, [3, 9, 8]) == 3
    A = [0, 1, 2, 3, 9, 8, 7]
    assert misc.get_subarray_start_ndx(A, [999]) == None

    with pytest.raises(ValueError) as e_info:
        A = [0, 1, 2, 3, 9, 8, 7]
        misc.get_subarray_start_ndx([1, 2], A)
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
    A = [1, 3, 2, 4]   # coverage: dtype
    val = misc.get_sorted_array(A)
    assert np.all(val[0] == [1, 2, 3, 4])
    assert np.all(val[1] == [0, 2, 1, 3])
    return


def test_get_ranked_array():
    A = np.array([1, 3, 2, 4])
    val = misc.get_ranked_array(A, reverse=False, verbose=False)
    assert (val[0] == [4, 3, 2, 1]).all()
    assert (val[1] == [3, 1, 2, 0]).all()

    A = [1, 3, 2, 4]   # coverage: dtype
    val = misc.get_ranked_array(A, reverse=True, verbose=True)
    assert np.all(val[0] == [1, 2, 3, 4])
    assert np.all(val[1] == [0, 2, 1, 3])
    return


def test_get_percentile():
    data = list(range(10))
    assert misc.get_percentile(data, p=50) == 4.5
    assert misc.get_percentile(data, p=80) == 7.2
    return


def test_get_quantile():
    data = list(range(10))
    assert misc.get_quantile(data, p=0.5) == 4.5
    assert misc.get_quantile(data, p=0.8) == 7.2
    return


def test_autodetect_header():
    assert misc.autodetect_header(pre3 + "1l2y.pdb") == 145
    assert misc.autodetect_header(f"{pre}/files/header_file1.txt") == 8
    assert misc.autodetect_header(f"{pre}/files/header_file2.txt") == 10
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

    # coverage
    cfg2()
    temp = cfg2.deepcopy()
    temp = cfg2.values()
    temp = cfg2.update_config({})

    default = {"colors": ["g", "r"]}
    alias_dict = {"color_positive": "teal",
                  "color_negative": "orange"}
    cfg3 = misc.CONFIG(default, **alias_dict)
    cfg3.update_by_alias(alias="color_positive", key="colors", key_ndx=0, **alias_dict)
    cfg3.update_by_alias(alias="color_negative", key="colors", key_ndx=1, **alias_dict)
    assert cfg3.colors[0] == "teal"
    assert cfg3.colors[1] == "orange"
    return


################################################################################
################################################################################
### coverage of less important functions


def test_apply_matplotlib_rc_settings():
    misc.apply_matplotlib_rc_settings()
    return


def test_HiddenPrints_ALL():
    with misc.HiddenPrints_ALL():
        print("will be hidden")
    return


def test_timeit():
    t = misc.timeit()
    t = misc.timeit(t)
    return


def test_cwd():
    misc.cwd()
    return


def test_pwd():
    misc.pwd()
    return


def test_pathexists():
    misc.pathexists(".")
    return


def test_dirpath():
    misc.dirpath(".", realpath=False)
    return


def test_joinpath():
    misc.joinpath(".", ".", realpath=False, create_dir=False)
    misc.joinpath(".", "./", realpath=False, create_dir=False)
    return


def test_rm():
    misc.rm("./temp.txt", pattern="./temp.txt")
    return


def test_bash_cmd():
    misc.bash_cmd("echo", verbose=True)
    return


def test_convert_image():
    image = pre2 + "TSNE_n10_v2.png"
    misc.convert_image(fin=image, fout="./temp.png")
    misc.rm("./temp.png")
    return


def test_convert_multiple_images():
    misc.convert_multiple_images(folder_in=pre2, folder_out=pre2, format="tiff")
    misc.rm(pre2, pattern="*tiff")
    return


def test_cprint():
    misc.cprint("text messsage", cprint_color="red")
    with pytest.raises(TypeError) as e_info:
        misc.cprint(0)
    return


def test_get_python_version():
    misc.get_python_version()
    return


def test_percent():
    assert misc.percent(5, 0) == 0.0
    return


def test_round_to_base():
    assert misc.round_to_base(2, base=5) == 0
    assert misc.round_to_base(3, base=5) == 5
    return


def test_norm_array():
    misc.norm_array([0, 1, 2])
    misc.norm_array(np.array([0, 1, 2]))
    return


def test_print_table():
    # coverage
    misc.print_table([[0, 1], [0, 1]], dtype=float)
    for i in range(1, 12):
        DATA = [[[0, 1] for _ in range(50)] for _ in range(i)]
        if i <= 10:
            misc.print_table(DATA)
        else:
            with pytest.raises(misc.Error) as e_info:
                misc.print_table(DATA)
    return


def test_save_table():
    xdata = [0, 1, 2, 3]
    with pytest.raises(TypeError):
        misc.save_table(data=[xdata, xdata], filename="")
    with pytest.raises(IndexError):
        misc.save_table(data=[xdata, xdata[:2]], save_as="./temp")   # unequal length

    misc.save_table(data=[xdata, xdata], filename="./temp", header="#header line")   # append .log
    misc.rm("./temp.log")
    return


@patch("matplotlib.pyplot.show")
def test_set_pad(mock_show):
    fig, ax = misc.figure()
    misc.set_pad(fig)
    misc.set_pad([ax, ax])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_legend(mock_show):
    fig, ax = misc.figure()
    plt.plot([0, 1], [0, 1], color="red")
    plt.plot([2, 3], [2, 3], color="blue")
    misc.legend()
    misc.legend(labels=["0", "1"], handlecolors=["red", "blue"])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_savefig(mock_show):
    fig, ax = misc.figure()
    misc.savefig(filename=None)
    return


def test_autoapply_limits():
    fig, ax = misc.figure()
    misc.autoapply_limits(fig)
    plt.plot([0, 1, 2], [0, 1, 2])
    misc.autoapply_limits(ax)

    obj = misc.pickle_load(pre4 + "RMSD_PLOT.pickle")
    misc.autoapply_limits(obj)
    plt.close("all")
    return


def test_hide_plot():
    fig, ax = misc.figure()
    misc.hide_plot(fig)
    fig, ax = misc.figure()
    misc.hide_plot(num=fig.number)
    return


@patch("matplotlib.pyplot.show")
def test_pickle_load(mock_show):
    obj = misc.pickle_load(pre4 + "RMSD_HIST.pickle")
    obj = misc.pickle_load(pre4 + "RMSD_PLOT.pickle")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_pickle_dump(mock_show):
    fig, ax = misc.figure()
    misc.pickle_dump(fig, save_as="./temp")        # appends .pickle
    misc.pickle_dump(fig, save_as="./temp")        # overwrite
    misc.pickle_dump([0, 1, 2], save_as="./temp")  # 'pickle.dumped object' instead of 'pickle.dumped figure'
    misc.rm("./temp.pickle")
    plt.close("all")

    # coverage
    with pytest.raises(TypeError) as e_info:
        misc.pickle_dump(fig)
    return


@patch("matplotlib.pyplot.show")
def test_pickle_plot(mock_show):
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"], import_settings=False, xscale="linear", yscale="linear")   # coverage
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"], save_as="./temp.png")

    # coverage
    with pytest.raises(TypeError) as e_info:
        fig, ax = misc.pickle_plot()

    misc.rm("./temp.png")
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_align_limits(mock_show):
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"])
    misc.align_limits(ax[0], ax[1], apply_on="xy", new_lim=[])
    misc.align_limits(ax[0], ax[1], apply_on="xy", new_lim=[0, 1])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_align_ticks(mock_show):
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"])
    misc.align_ticks(ax[0], ax[1], apply_on="xy", new_ticks=[])
    misc.align_ticks(ax[0], ax[1], apply_on="xy", new_ticks=[0, 1])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_align_ticklabels(mock_show):
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"])
    misc.align_ticklabels(ax[0], ax[1], apply_on="xy", new_ticklabels=[])
    misc.align_ticklabels(ax[0], ax[1], apply_on="xy", new_ticklabels=[0, 1])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_apply_shared_axes(mock_show):
    fig, ax = misc.pickle_plot([pre4 + "RMSD_PLOT.pickle", pre4 + "RMSD_HIST.pickle"])
    # coverage
    misc.apply_shared_axes(ax, grid=[2, 1])
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_convert_ticklabels(mock_show):
    fig, ax = misc.figure(grid=[2, 1])
    plt.sca(ax[0])
    plt.plot(list(range(100)), list(range(100)))
    plt.sca(ax[1])
    plt.plot(list(range(100)), list(range(100)))
    misc.convert_ticklabels(ax, multiplier=10, apply_on="xy", prec=1)
    misc.convert_ticklabels(ax, multiplier=10, apply_on="xy", prec=0)
    plt.close("all")
    return


def test_number_base_factorization():
    x = misc.number_base_factorization(123)
    x()
    x = misc.number_base_factorization(0.123)
    x()
    return


def test_setup_ticks():
    majorticks, minorticks = misc.setup_ticks(vmin=0, vmax=10, major_base=5, minor_base=None)
    majorticks, minorticks = misc.setup_ticks(vmin=0, vmax=10, major_base=5, minor_base=1)
    return


def test_setup_logscale_ticks():
    majorticks, minorticks = misc.setup_logscale_ticks(vmax=100)
    return


@patch("matplotlib.pyplot.show")
def test_set_logscale_ticks(mock_show):
    fig, ax = misc.figure()
    plt.plot(list(range(100)), list(range(100)))
    misc.set_logscale_ticks(ax, apply_on="xy", vmax=None, minorticks=True)
    misc.set_logscale_ticks(ax, apply_on="xy", vmax=20, minorticks=False)
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_create_cmap(mock_show):
    seq = ["lightblue", 2/6, "lightgreen", 3/6, "yellow", 4/6, "orange", 5/6, "red"]
    cmap = misc.create_cmap(seq, vmin=0, vmax=12)

    # coverage
    fig, ax = misc.figure()
    cmap = misc.create_cmap(seq, ax=ax)
    plt.close("all")
    return


@patch("matplotlib.pyplot.show")
def test_add_cbar_ax(mock_show):
    fig, ax = misc.figure()
    # coverage
    cbar_ax = misc.add_cbar_ax(ax, bounds=[0, 0, 1, 0.05])
    cbar_ax = misc.add_cbar_ax(ax, location="right", orientation="horizontal")
    cbar_ax = misc.add_cbar_ax(ax, location="left", orientation="horizontal")
    cbar_ax = misc.add_cbar_ax(ax, location="top", orientation="vertical")
    cbar_ax = misc.add_cbar_ax(ax, location="bottom", orientation="vertical")
    plt.close("all")
    return
