"""Helpers for coordinating parallel Methods using MPI."""

import numpy as np
from time import time

from mpi4py import MPI  # calls MPI_Init()

from numpy.typing import ArrayLike


class ParallelHelper:
    """
    Helper for coordinating parallel Methods.

    Parameters
    ----------
    vector_size : int
        The size of the vectors to be distributed among processors.
    """

    # Shared class attributes
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _root = 0
    _is_root = _rank == _root
    _parallel = _size > 1

    def __init__(self, shape: int | tuple[int, ...]) -> None:
        # Following values set upon calling `_setup_job_map`.
        self._imin: int  # Minimum index for this processor
        self._imax: int  # Maximum index for this processor

        if isinstance(shape, int):
            shape = (shape,)
        self._bufshape: tuple[int, ...] = shape
        self._bufsize: int = np.prod(shape)

        if self._size > shape[0]:
            raise ValueError(
                f"Number of processors ({self._size}) is too many for this problem; "
                f"expected at most {shape[0]}"
            )
        self._setup_job_map(shape[0])  # Assume first dimension is the one to distribute

        self._recvbuf: np.array | None = None  # Receive buffer for reductions

    @property
    def comm(self) -> MPI.Intracomm:
        """MPI communicator."""
        return self._comm

    @property
    def rank(self) -> int:
        """Rank of this processor."""
        return self._rank

    @property
    def size(self) -> int:
        """Total number of processors."""
        return self._size

    @property
    def root(self) -> int:
        """Root processor rank."""
        return self._root

    @property
    def is_root(self) -> bool:
        """Whether this is the parent processor."""
        return self._is_root

    @property
    def parallel(self) -> bool:
        """Whether running in parallel mode."""
        return self._parallel

    @property
    def imin(self) -> int:
        """Minimum index for this processor."""
        return self._imin

    @property
    def imax(self) -> int:
        """Maximum index for this processor."""
        return self._imax

    @property
    def bufshape(self) -> tuple[int, ...]:
        """Shape of the buffers for this processor."""
        return self._bufshape

    @property
    def bufsize(self) -> int:
        """Buffer size for this processor."""
        return self._bufsize

    def _setup_job_map(self, nelem: int) -> None:
        """
        Distribute blocks of indicies among processes given the number of array elements.

        Additionally, if we are the root, also report the range of
        indicies each processor will work on.

        Parameters
        ----------
        nelem : int
            The number of elements in the vector. Usually the number
            of determinants.
        """
        all_indicies = np.arange(nelem)
        proc_indicies = np.array_split(all_indicies, self._size)
        iproc_indicies = proc_indicies[self._rank]

        self.print(f"Generated vector indicies job map for size {nelem}:")
        self.print(f"{'iproc':>8} {'min(index)':>12} {'max(index)':>12}")

        for iproc in range(self._size):
            self.print(
                f"{iproc:>8} "
                f"{min(proc_indicies[iproc]):>12d} "
                f"{max(proc_indicies[iproc]) + 1:>12d} "
            )

            if iproc == self._rank:
                assert np.array_equal(iproc_indicies, proc_indicies[iproc])

        self._imin = int(min(iproc_indicies))
        self._imax = int(max(iproc_indicies)) + 1  # +1 for python indexing

        return

    def get_rng_seed(self, rng_seed: None | int | ArrayLike = None) -> int:
        """
        Work out the rng seed for the calling processor.

        If on the root, also report the rng seeds for all processors.

        Parameters
        ----------
        rng_seed : int or array_like of ints
            The user provided initial rng seed.

        Returns
        -------
        seed : int or array_like of ints
            The rng seed(s) to use for the NumPy generator, which should be
            rank and calculation safe.
        """
        if rng_seed is None:
            rng_seed = int(time())
        my_seed = rng_seed + self._size * self._rank

        self.print("Setting processor rng seeds to:")
        self.print(f"{'iproc':>8} {'seed'}")
        for iproc in range(self._size):
            iseed = rng_seed + self._size * iproc

            if iproc == self._rank:
                assert iseed == my_seed

            self.print(f"{iproc:>8d} {iseed}")

        return my_seed

    def print(self, text: str = "") -> None:
        """
        Print text only from the parent processor.

        Parameters
        ----------
        text : str
            The text to print.
        """
        if self._is_root:
            print(text)
        return

    def barrier(self) -> None:
        """Wrap MPI Barrier."""
        self._comm.Barrier()
        return

    def abort(self, errorcode: int) -> None:
        """
        Abort the MPI process.

        Parameters
        ----------
        errorcode : int
            The error code to return upon aborting.
        """
        self._comm.Abort(errorcode=errorcode)
        return

    def broadcast(self, array: np.array) -> np.array:
        """Broadcast an array from the root processor.

        Parameters
        ----------
        array : np.array
            The array to be broadcast. On the root processor,
            this contains the data to be sent. On other processors,
            this will be updated with the broadcast data.

        Returns
        -------
        array : np.array
            The broadcasted array. All processors will have
            the same data after the broadcast.
        """
        self._comm.Bcast(array, root=self._root)

        return array

    def allocate_reduce_buffers(self) -> None:
        """Allocate send and receive buffers for communication."""
        if self._recvbuf is None:
            self.print(f"Allocating one recv buffer with shape {self._bufshape}.")
            self._recvbuf = np.zeros(self._bufshape, dtype=float)
        else:
            self.print("Buffers already allocated; skipping.")

    def allreduce_sum(self, array: np.array) -> np.array:
        """
        Allreduce an array across all processors.

        Parameters
        ----------
        array : np.array
            The array to be reduced. All processors will be
            updated with the reduced values.
            It must have the same shape as the `bufshape` attribute.

        Returns
        -------
        array : np.array
            The reduced array. All processors will contain
            the reduced values.
        """
        if self._recvbuf is None:
            raise RuntimeError("Buffers not allocated.")

        if array.shape != self._recvbuf.shape:
            raise ValueError(
                f"Array shape ({array.shape}) does not match buffer shape ({self._recvbuf.shape})."
            )

        self._comm.Allreduce(
            array,
            self._recvbuf,
            op=MPI.SUM,
        )

        array = self._recvbuf.copy()
        return array
