"""Helpers for coordinating parallel Methods using MPI."""

import numpy as np
from time import time

from mpi4py import MPI  # calls MPI_Init()

from typing import Callable, Any
from numpy.typing import ArrayLike, NDArray as Array


class ParallelHelper:
    """
    Helper for coordinating parallel Methods.

    Parameters
    ----------
    shape : int or tuple of ints
        The size of the arrays to be distributed among processors.
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
        self._nrows: int  # Number of rows stored on this processor
        self._rank_index_bounds: dict[int, tuple[int, int]] = {}  # Indices for all ranks
        self._index_to_rank: np.ndarray = np.empty(shape[0], dtype=int)  # Host rank for each global index

        if isinstance(shape, int):
            shape = (shape,)
        self._bufshape: tuple[int, ...] = shape
        self._bufsize: int = np.prod(shape)

        # Distribute along the first dimension
        if self._size > shape[0]:
            raise ValueError(
                f"Number of processors ({self._size}) is too many for this problem; "
                f"expected at most {shape[0]}"
            )
        self._setup_job_map(shape[0])

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
    def nrows(self) -> int:
        """
        Number of rows stored on this processor.

        In a fully assembled array, these rows correspond to `imin` through `imax`.
        """
        return self._nrows

    @property
    def rank_index_bounds(self) -> dict[int, tuple[int, int]]:
        """Start and end index bounds for each rank."""
        return self._rank_index_bounds

    @property
    def index_to_rank(self) -> np.ndarray:
        """Array mapping each row index to the rank that owns it."""
        return self._index_to_rank

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

        self._rank_index_bounds = {}
        for iproc in range(self._size):
            indices = proc_indicies[iproc]
            if len(indices) == 0:
                start, end = 0, 0
            else:
                start = int(indices[0])
                end = int(indices[-1]) + 1
            self._rank_index_bounds[iproc] = (start, end)
            self._index_to_rank[start:end] = iproc

        self.print(f"Generated vector indicies job map for size {nelem}:")
        self.print(f"{'iproc':>8} {'min(index)':>12} {'max(index)':>12}")

        for iproc in range(self._size):
            start, end = self._rank_index_bounds[iproc]
            self.print(
                f"{iproc:>8} "
                f"{start:>12d} "
                f"{end:>12d} "
            )

            if iproc == self._rank:
                assert np.array_equal(iproc_indicies, proc_indicies[iproc])

        self._imin = self._rank_index_bounds[self._rank][0]
        self._imax = self._rank_index_bounds[self._rank][1]
        self._nrows = self._imax - self._imin

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

    def safe_noncollective(
        self, func: Callable[[...], Array[tuple[int, int]]], *args, **kwargs
    ) -> Any:
        """
        Run a function that returns a 2D array on the root rank in a deadlock safe way.

        If an exception is raised on the active rank then this is caught and
        raised collectively.

        Parameters
        ----------
        func: function
            The operation to be performed on a single rank. It must return a 2D array
            when working correctly.
        *args, **kwargs:
            Arguments for the function.

        Returns
        -------
        Any
            The result of ``func``, broadcasted to all ranks.

        """
        # Thanks to Connor J Ward of the Firedrake project for suggesting this function!
        result = None
        is_array = np.array([False], dtype=np.bool)
        arr_shape = np.array((0, 0), dtype=int)

        if self._is_root:
            try:
                self.print(func)
                result = func(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    is_array[0] = True
                    arr_shape[:] = result.shape
            except BaseException as e:
                result = e

        is_array = self.broadcast(is_array)
        arr_shape = self.broadcast(arr_shape)

        if is_array:
            if result is None:  # prep non-root rank for broadcast
                result = np.empty(arr_shape, dtype=float)
            result = self.broadcast(result)
        else:
            # Use the generic pickle-based broadcast (slower)
            result = self._comm.bcast(result, root=self._root)

        if isinstance(result, BaseException):
            raise result
        else:
            return result

    def broadcast(self, array: np.array) -> np.array:
        """
        Broadcast an array from the root processor.

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

    def gather_rows(self, array: np.ndarray) -> np.ndarray | None:
        """Gather row-partitioned arrays on the root processor.

        The first dimension must contain this rank's rows, while all remaining
        dimensions must match the configured global buffer shape. Row blocks
        are gathered in rank order, which is also global row order.
        """
        expected_shape = (self._nrows,) + self._bufshape[1:]
        if array.shape != expected_shape:
            raise ValueError(
                f"Array shape ({array.shape}) does not match local row shape "
                f"({expected_shape})."
            )

        item_size = int(np.prod(self._bufshape[1:], dtype=int))
        counts = np.array(
            [end - start for start, end in self._rank_index_bounds.values()],
            dtype=np.int32,
        )
        counts *= item_size
        displacements = np.zeros(self._size, dtype=np.int32)
        displacements[1:] = np.cumsum(counts[:-1])

        mpi_type = MPI._typedict[np.dtype(array.dtype).char]

        if self._is_root:
            gathered = np.empty(self._bufsize, dtype=array.dtype)
            receive_buffer = [gathered, counts, displacements, mpi_type]
        else:
            gathered = None
            receive_buffer = None

        self._comm.Gatherv(array.ravel(), receive_buffer, root=self._root)

        if self._is_root:
            return gathered.reshape(self._bufshape)
        return None

    def update_matrix_element(self, value: float,  from_idx: tuple[int, int], to_idx: tuple[int, int]):
        """
        ...

        Parameters
        ----------
        value : float
            Quantity to be added to the existing element at [index0, index1].
        index0, index1 : int
            Indicies for the matrix element to be updated.
        """
        pass
        # If I'm the sending rank, I need to look up who to send to

        # but the receiving rank must also know who to receive from

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
        array = np.asarray(array)
        reduced = np.empty_like(array)
        self._comm.Allreduce(
            array,
            reduced,
            op=MPI.SUM,
        )
        return reduced

# TODO write gather routine for fragments of e.g. density matrix