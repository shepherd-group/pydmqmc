import pickle
import numpy as np
from pytest import fixture
from os.path import join, exists

from pydmqmc.utils import save_array, save_report

class TestSave():
    """Test save_array."""

    @fixture(autouse=True)
    def _setup(self, tmp_path_factory):
        self._tmpdir = tmp_path_factory.mktemp("test_save")

    def test_save_array_csv(self):
        arr = np.arange(10)
        save_array(arr,
                   join(self._tmpdir, "array"),
                   filetype="csv")

        expected_file = join(self._tmpdir, f"array.csv")
        assert exists(expected_file)
        loaded = np.loadtxt(expected_file)
        assert np.array_equal(arr, loaded)

    def test_save_array_txt(self):
        arr = np.arange(10)
        save_array(arr,
                   join(self._tmpdir, "array"),
                   filetype="txt")

        expected_file = join(self._tmpdir, f"array.txt")
        assert exists(expected_file)
        loaded = np.loadtxt(expected_file)
        assert np.array_equal(arr, loaded)

    def test_save_array_npy(self):
        arr = np.arange(10)
        save_array(arr,
                   join(self._tmpdir, "array"),
                   filetype="npy")

        expected_file = join(self._tmpdir, f"array.npy")
        assert exists(expected_file)
        loaded = np.load(expected_file)
        assert np.array_equal(arr, loaded)

    def test_save_array_pkl(self):
        arr = np.arange(10)
        save_array(arr,
                   join(self._tmpdir, "array"),
                   filetype="pkl")

        expected_file = join(self._tmpdir, f"array.pkl")
        assert exists(expected_file)
        with open(expected_file, "rb") as f:
            loaded = pickle.load(f)
        assert np.array_equal(arr, loaded)

class TestSaveReport():
    """Test save_report."""

    @fixture(autouse=True)
    def _setup(self, tmp_path_factory):
        self._tmpdir = tmp_path_factory.mktemp("test_save_report")
        self._data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    def test_save_report_csv(self):
        save_report(self._data,
                    join(self._tmpdir, "report"),
                    index_col='a',
                    filetype="csv")

        expected_file = join(self._tmpdir, f"report.csv")
        assert exists(expected_file)
        loaded = np.genfromtxt(expected_file, delimiter=',', names=True)
        assert loaded.shape[0] == 2
        assert loaded['a'][0] == 1
        assert loaded['b'][1] == 4

    def test_save_report_txt(self):
        save_report(self._data,
                    join(self._tmpdir, "report"),
                    index_col='a',
                    filetype="txt")

        expected_file = join(self._tmpdir, f"report.txt")
        assert exists(expected_file)
        loaded = np.genfromtxt(expected_file, delimiter='\t', names=True)
        assert loaded.shape[0] == 2
        assert loaded['a'][0] == 1
        assert loaded['b'][1] == 4

    def test_save_report_pkl(self):
        save_report(self._data,
                    join(self._tmpdir, "report"),
                    index_col='a',
                    filetype="pkl")

        expected_file = join(self._tmpdir, f"report.pkl")
        assert exists(expected_file)
        with open(expected_file, "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == 2
        assert loaded[0]['a'] == 1
        assert loaded[1]['b'] == 4