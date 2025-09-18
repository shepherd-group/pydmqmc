"""Utilities for saving data."""

import csv
import pickle
import numpy as np

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
    """
    if data is None:
        raise RuntimeError(
            "Provided data is None! Did you remember to run the simulation?"
        )

    if filetype == "txt":
        np.savetxt(basename + ".txt", data)
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
    Save a dictionary of data as a chosen file type.

    Parameters
    ----------
    list_of_dicts : list of dictionaries with strings as keys.
        Data to be written to disk. Each dictionary must have
        identical keys.
    index_col : string, optional
        Name of the column to put first. Must be a key in the
        dictionaries in `list_of_dicts`.
    basename : str
        Base filename (i.e., without extension) to which to write data.
    filetype : str, default "csv"
        File type (aka extension) with which to save `list_of_dicts`.
        Supported types are:

        - "csv" : comma-separated value file
        - "txt" : text file (space-delimited)
        - "pkl" : pickle file

    pickle_protocol : unt, optional
        Protocol version to use with pickle. If none, uses `pickle`'s default.
    """
    if list_of_dicts is None:
        raise RuntimeError(
            "Provided data is None! Did you remember to run the simulation?"
        )

    if filetype == "txt" or filetype == "csv":
        filename = basename + "." + filetype

        # Create a list of field names from the dictionaries in
        # list_of_dicts but with index_col first
        fields = list(list_of_dicts[0].keys())
        fields.remove(index_col)
        fields = [index_col] + fields

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            writer.writeheader()
            writer.writerows(list_of_dicts)

    elif filetype == "pkl":
        filename = basename + ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(list_of_dicts, f, protocol=pickle_protocol)
    else:
        raise RuntimeError(f"File type {filetype} is not recognized!")
