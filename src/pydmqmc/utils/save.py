"""Utilities for saving data."""

import csv
import pickle
import numpy as np

import warnings

from numpy.typing import ArrayLike


def save_array(
    data: ArrayLike,
    basename: str,
    filetype: str = "csv",
    pickle_protocol: int | None = None,
) -> None:
    """
    Save an array as a chosen file type.

    Parameters
    ----------
    data : array_like
        Data to be written to disk.
    basename : str
        Base filename (i.e., without extension) to which to write data.
    filetype : str, default "csv"
        File type (aka extension) with which to save the data.
        Supported types are:

        - "csv" : comma-separated value file
        - "npy" : NumPy binary file
        - "pkl" : Python pickle file
        - "txt" : text file (space-delimited)

    pickle_protocol : unt, optional
        Protocol version to use with pickle. If none, uses `pickle`'s default.

    See Also
    --------
    save_report : Save lists of dictionaries as a chosen file type.
    """
    if data is None:
        raise RuntimeError(
            "Provided data is None! Did you remember to run the simulation?"
        )

    if filetype == "txt":
        # delimit with tabs instead of spaces
        # to match the txt dialect in save_report
        np.savetxt(basename + ".txt", data, delimiter="\t")
    elif filetype == "csv":
        np.savetxt(basename + ".csv", data, delimiter=",")
    elif filetype == "npy":
        np.save(basename + ".npy", data)
    elif filetype == "pkl":
        filename = basename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle_protocol)
    else:
        raise RuntimeError(f"File type {filetype} is not recognized!")


def save_report(
    list_of_dicts: list[dict],
    basename: str,
    index_col: str | None = None,
    filetype: str = "csv",
    pickle_protocol: int | None = None,
) -> None:
    """
    Save a lists of dictionaries as a chosen file type.

    Parameters
    ----------
    list_of_dicts : list of dictionaries with strings as keys.
        Data to be written to disk. Each dictionary must have
        identical keys.
    basename : str
        Base filename (i.e., without extension) to which to write data.
    index_col : string, optional
        For "csv" or "txt" filetypes, name of the column to put first.
        Must be a key in the dictionaries in `list_of_dicts`.
        If `index_col` is not supplied, the first column saved to file
        will be arbitrary.
    filetype : str, default "csv"
        File type (aka extension) with which to save `list_of_dicts`.
        Supported types are:

        - "csv" : comma-separated value file
        - "txt" : text file (space-delimited)
        - "pkl" : pickle file

    pickle_protocol : unt, optional
        Protocol version to use with pickle. If none, uses `pickle`'s default.

    See Also
    --------
    save_array : Save an array as a chosen file type.
    """
    if not list_of_dicts:
        raise RuntimeError("No data provided! Did you remember to run the simulation?")

    if filetype == "txt" or filetype == "csv":
        filename = basename + "." + filetype

        # Create a list of field names from the dictionaries
        fields = list(list_of_dicts[0].keys())
        if index_col is not None:
            # Put index_col first in the list
            fields.remove(index_col)
            fields = [index_col] + fields

        if filetype == "csv":
            dialect = "excel"  # default dialect
        else:
            dialect = "excel-tab"  # use tabs to deliminate columns

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, dialect=dialect)

            writer.writeheader()
            writer.writerows(list_of_dicts)

    elif filetype == "pkl":
        if index_col is not None:
            warnings.warn("Parameter index_col is ignored with 'pkl' filetype.")
        filename = basename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(list_of_dicts, f, protocol=pickle_protocol)
    else:
        raise RuntimeError(f"File type {filetype} is not recognized!")
