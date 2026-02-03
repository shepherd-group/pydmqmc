"""Helpers for coordinating parallel Methods using MPI."""

import numpy as np
import functools

from mpi4py import MPI  # calls MPI_Init()

from numpy.typing import ArrayLike


def abort_on_exception(func):
    """Aborts MPI process if wrapped function raises an exception."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.print(f"Exception in {func.__name__}: {e}")
            self.abort(1)

    return wrapper


class ParallelHelper:
    """
    Helper for coordinating parallel Methods.

    Parameters
    ----------
    vector_size : int
        The size of the vectors to be distributed among processors.
    initiator_threshold : int
        The initiator threshold for the calculation. If greater
        than zero, larger buffers are allocated to accommodate
        initiator-specific data.
    """

    # Shared class attributes
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _root = 0
    _parent = _rank == _root
    _parallel = _size > 1

    def __init__(self, vector_size: int, initiator_threshold: int) -> None:
        # Following values set upon calling `_setup_job_map`.
        self._imin: int  # Minimum index for this processor
        self._imax: int  # Maximum index for this processor
        self._vecsize: int = vector_size
        self._bufshape: tuple[int, int]

        if initiator_threshold > 0:
            self._bufshape = (5, vector_size)
        else:
            self._bufshape = (1, vector_size)

        self._setup_job_map(vector_size)

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
    def parent(self) -> bool:
        """Whether this is the parent processor."""
        return self._parent

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
    def bufshape(self) -> tuple[int, int]:
        """Buffer shape for this processor."""
        return self._bufshape

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

    def allocate_buffers(self) -> None:
        """Allocate send and receive buffers for communication."""
        if self._recvbuf is None:
            self.print(f"Allocating one recv buffer with size {self._bufshape}.")
            self._recvbuf = np.zeros(self._bufshape, dtype=float)
        else:
            self.print("Buffers already allocated; skipping.")

    def get_rng_seed(self, rng_seed: int | ArrayLike) -> int:
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
        seed = rng_seed + self._size * self._rank

        self.print("Setting processor rng seeds to:")
        self.print(f"{'iproc':>8} {'seed'}")
        for iproc in range(self._size):
            iseed = rng_seed + self._size * iproc

            if iproc == self._rank:
                assert iseed == seed

            self.print(f"{iproc:>8d} {iseed}")

        return seed

    def print(self, text: str = "") -> None:
        """
        Print text only from the parent processor.

        Parameters
        ----------
        text : str
            The text to print.
        """
        if self._parent:
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

    def bcast(self, array: np.array) -> np.array:
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

    def sum_reduce(self, array: np.array) -> np.array:
        """
        Reduce an array across all processors.

        Parameters
        ----------
        array : np.array
            The array to be reduced. On the parent processor,
            this will be updated with the reduced values.
            It must have the same size as the `bufsize` attribute.

        Returns
        -------
        array : np.array
            The reduced array. On the parent processor,
            this will contain the reduced values. Other
            processors will retain the original values.
        """
        if self._recvbuf is None:
            raise RuntimeError("Buffers not allocated.")

        if array.size != self._recvbuf.size:
            raise ValueError("Array size does not match buffer size.")

        self._comm.Reduce(
            array,
            self._recvbuf,
            op=MPI.SUM,
            root=self._root,
        )

        if self._parent:
            array = self._recvbuf.copy()
            print(array)
        return array


# NOTE: The following class does not yet have tests as it is currently unused
class TwoMatrixParallelHelper(ParallelHelper):
    """
    Helper for coordinating parallel Methods that involve two matrices.

    These matrices will be internally named p and q and will share
    send and receive buffers, `pq_sendbuf` and `pq_recvbuf`. Their derivatives,
    dp and dq, will also share send and receive buffers, `dpdq_sendbuf` and
    `dpdq_recvbuf`. The derivative buffers may be larger depending on the
    value of `initiator_threshold`.

    Parameters
    ----------
    vector_size : int
        The size of the vectors to be distributed among processors.
    initiator_threshold : int
        The initiator threshold for the calculation. If greater
        than zero, larger buffers are allocated to accommodate
        initiator-specific data.
    """

    def __init__(self, vector_size: int, initiator_threshold: int):
        super().__init__(vector_size)

        self._pq_sendbuf: np.array | None = None
        self._pq_recvbuf: np.array | None = None
        self._dpdq_sendbuf: np.array | None = None
        self._dpdq_recvbuf: np.array | None = None

        self._buffers = [
            self._pq_sendbuf,
            self._pq_recvbuf,
            self._dpdq_sendbuf,
            self._dpdq_recvbuf,
        ]

        self._pq_bufshape = (2 * self._vecsize,)
        if initiator_threshold > 0:
            self._dpdq_bufshape = (5, 2 * self._vecsize)
        else:
            self._dpdq_bufshape = (1, 2 * self._vecsize)

    @property
    def pq_bufshape(self) -> tuple[int, int]:
        """Buffer shape for p and q arrays."""
        return self._pq_bufshape

    @property
    def dpdq_bufshape(self) -> tuple[int, int]:
        """Buffer shape for dp and dq arrays."""
        return self._dpdq_bufshape

    def allocate_buffers(self) -> None:
        """Allocate send and receive buffers for communication.

        Parameters
        ----------
        initiator_threshold : int
            The initiator threshold for the calculation. If greater
            than zero, larger buffers are allocated to accommodate
            initiator-specific data.
        """
        if None in self._buffers:
            self.print(
                "Allocating two send and two additional recv buffers with sizes \n"
                f"{self._pq_bufshape}, {self._dpdq_bufshape} and "
                f"{self._pq_bufshape}, {self._dpdq_bufshape} respectively."
            )

            self._pq_sendbuf = np.zeros(self._pq_bufshape, dtype=float)
            self._pq_recvbuf = np.zeros(self._pq_bufshape, dtype=float)
            self._dpdq_sendbuf = np.zeros(self._dpdq_bufshape, dtype=float)
            self._dpdq_recvbuf = np.zeros(self._dpdq_bufshape, dtype=float)
        else:
            self.print("Buffers already allocated; skipping.")

    def pq_bcast(
        self,
        p: np.array,
        q: np.array,
    ) -> tuple[np.array, np.array]:
        """Broadcast two arrays from the root processor."""
        if None in self._buffers:
            raise RuntimeError("Buffers not allocated.")

        if self._parent:
            self._pq_sendbuf[: self._bufshape] = p[:, 0]
            self._pq_sendbuf[self._bufshape :] = q[:, 0]

        self.bcast(self._pq_sendbuf)

        p[:, 0] = self._pq_sendbuf[: self._bufshape]
        q[:, 0] = self._pq_sendbuf[self._bufshape :]

        return p, q

    def dpdq_reduce(
        self,
        dp: np.array,
        dq: np.array,
    ) -> tuple[np.array, np.array]:
        """Reduce two arrays across all processors."""
        if None in self._buffers:
            raise RuntimeError("Buffers not allocated.")

        self.dpdq_sendbuf[:, : self._bufshape] = dp[:, :]
        self.dpdq_sendbuf[:, self._bufshape :] = dq[:, :]

        self._comm.Reduce(
            self.dpdq_sendbuf,
            self.dpdq_recvbuf,
            op=MPI.SUM,
            root=self._root,
        )

        if self._parent:
            dp[:, :] = self.dpdq_recvbuf[:, : self._bufshape]
            dq[:, :] = self.dpdq_recvbuf[:, self._bufshape :]

        return dp, dq

    def pq_allreduce(
        self,
        p: np.array,
        q: np.array,
    ) -> tuple[np.array, np.array]:
        """Allreduce two arrays across all processors."""
        if None in self._buffers:
            raise RuntimeError("Buffers not allocated.")

        self._pq_sendbuf[: self._bufshape] = p[:, 0]
        self._pq_sendbuf[self._bufshape :] = q[:, 0]

        self._comm.Allreduce(
            self._pq_sendbuf,
            self.pq_recvbuf,
            op=MPI.SUM,
        )

        p[:, 0] = self.pq_recvbuf[: self._bufshape]
        q[:, 0] = self.pq_recvbuf[self._bufshape :]

        return p, q
