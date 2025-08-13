"""Utilities for saving data."""

import numpy as np

from numpy.typing import ArrayLike


def save_data(data: ArrayLike,
              basename: str,
              filetype: str = "txt") -> None:
    """
    Save results of the calculation to a chosen file type.

    Parameters
    ----------
    data : array_like
        Data to be written to disk.
    basename : str
        Base filename (i.e., without extension) to which to write data.
    filetype : str, default "txt"
        File type (aka extension) with which to save the data.
        Supported types are:

        - "txt" : text file (space-delimited)
        - "csv" : comma-separated value file
        - "npy" : NumPy binary file
    """
    if data is None:
        raise RuntimeError("Provided data is None! Did you "
                           "remember to run the simulation?")

    if filetype == "txt":
        np.savetxt(basename+".txt", data)
    elif filetype == "csv":
        np.savetxt(basename+".csv", data, delimiter=',')
    elif filetype == "npy":
        np.save(basename+".npy", data)
    else:
        raise RuntimeError(f"File type {filetype} "
                            "is not recognized!")
