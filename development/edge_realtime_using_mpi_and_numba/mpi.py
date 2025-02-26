#!/usr/bin/env python

import numpy as np

from time import sleep
from mpi4py import MPI


class ParallelHelper:
    r''' TODO: Docstring here.
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = 0
    parent = rank == root
    parallel = size > 1

    def __init__(self) -> None:
        r''' TODO: Docstring here.
        '''
        # Values only set upon calling `setup_job_map`.
        # The minimum and maximum index for the calling processor.
        self.imin: int
        self.imax: int
        # The size of p or q, i.e. ndets.
        self.bufsize: int
        # A recv buffer with size (1, ndets,) or (5, ndets).
        self.recvbuf: np.array
        # A send buffer with size (2*ndets,).
        self.pq_sendbuf: np.array
        # A recv buffer with size (2*ndets,).
        self.pq_recvbuf: np.array
        # A send buffer with size (1, 2*ndets,) or (5, 2*ndets).
        self.dpdq_sendbuf: np.array
        # A recv buffer with size (1, 2*ndets,) or (5, 2*ndets).
        self.dpdq_recvbuf: np.array

        return

    def sleep(self) -> None:
        r''' TODO: Docstring here.
        '''
        sleep(self.rank)
        return

    def print(self, text: str = '') -> None:
        r''' TODO: Docstring here.
        '''
        if self.parent:
            print(text)
        return

    def barrier(self) -> None:
        r''' TODO: Docstring here.
        '''
        self.comm.Barrier()
        return

    def abort(self, errorcode: int) -> None:
        r''' TODO: Docstring here.
        '''
        self.comm.Abort(errorcode=errorcode)
        return

    def bcast(self, array: np.array) -> np.array:
        r''' TODO: Docstring here.
        '''
        self.comm.Bcast(array, root=self.root)

        return array

    def reduce(self, array: np.array) -> np.array:
        r''' TODO: Docstring here.
        '''
        assert array.shape == self.recvbuf.shape

        self.comm.Reduce(
            array,
            self.recvbuf,
            op=MPI.SUM,
            root=self.root,
        )

        if self.parent:
            array[:, :] = self.recvbuf[:, :]

        return array

    def pq_bcast(
            self,
            p: np.array,
            q: np.array,
        ) -> tuple[np.array, np.array]:
        r''' TODO: Docstring here.
        '''
        if self.parent:
            self.pq_sendbuf[:self.bufsize] = p[:, 0]
            self.pq_sendbuf[self.bufsize:] = q[:, 0]

        self.bcast(self.pq_sendbuf)

        p[:, 0] = self.pq_sendbuf[:self.bufsize]
        q[:, 0] = self.pq_sendbuf[self.bufsize:]

        return p, q

    def dpdq_reduce(
            self,
            dp: np.array,
            dq: np.array,
        ) -> tuple[np.array, np.array]:
        r''' TODO: Docstring here.
        '''
        self.dpdq_sendbuf[:, :self.bufsize] = dp[:, :]
        self.dpdq_sendbuf[:, self.bufsize:] = dq[:, :]

        self.comm.Reduce(
            self.dpdq_sendbuf,
            self.dpdq_recvbuf,
            op=MPI.SUM,
            root=self.root,
        )

        if self.parent:
            dp[:, :] = self.dpdq_recvbuf[:, :self.bufsize]
            dq[:, :] = self.dpdq_recvbuf[:, self.bufsize:]

        return dp, dq

    def pq_allreduce(
            self,
            p: np.array,
            q: np.array,
        ) -> tuple[np.array, np.array]:
        r''' TODO: Docstring here.
        '''
        self.pq_sendbuf[:self.bufsize] = p[:, 0]
        self.pq_sendbuf[self.bufsize:] = q[:, 0]

        self.comm.Allreduce(
            self.pq_sendbuf,
            self.pq_recvbuf,
            op=MPI.SUM,
        )

        p[:, 0] = self.pq_recvbuf[:self.bufsize]
        q[:, 0] = self.pq_recvbuf[self.bufsize:]

        return p, q

    def get_rng_seed(self, rng_seed: int, icalc: int) -> int:
        r''' Work out the rng seed for the calling processor. If on
        the root, also report the rng seeds for all processors.

        Parameters
        ----------
        rng_seed : int
            The user provided initial rng seed.
        icalc : int
            The current calculation number.

        Returns
        -------
        seed : int
            The rng seed to use for the NumPy generator, which should be
            rank and calculation safe.
        '''
        seed = rng_seed + self.size*icalc + self.rank

        self.print('Setting processor rng seeds to:')
        self.print(f'{"iproc":>8} {"seed":>24}')
        for iproc in range(self.size):
            iseed = rng_seed + self.size*icalc + iproc

            if iproc == self.rank:
                assert iseed == seed

            self.print(f'{iproc:>8d} {iseed:>24d}')

        return seed

    def setup_job_map(self, ndets: int, ninit: float) -> None:
        r''' Given the size for a vector, work out the blocks of indicies
        for each processor. From the calculated blocks of indicies, set
        the various relevant quantities for iterating and communication
        during the calculation.

        The allocated arrays are used in calling the various
        MPI routines, e.g. bcast, gather, etc...

        Additionally, if we are the root, also report the range of
        indicies each processor will work on.

        Parameters
        ----------
        ndets : int
            The number of elements in the wavefunction vector.
        ninit : float
            The initiator threshold, if greater than zero the
            buffers shape is increased to accomidate the iniatior
            information.
        '''
        self.bufsize = ndets

        pq_shape = (2*self.bufsize,)

        if ninit > 0.0:
            shape = (5, self.bufsize)
            dpdq_shape = (5, 2*self.bufsize)
        else:
            shape = (1, self.bufsize)
            dpdq_shape = (1, 2*self.bufsize)

        self.print(
            'Allocating two send and three recv buffers with sizes\n'
            f' {pq_shape}, {dpdq_shape} and '
            f'{shape}, {pq_shape}, {dpdq_shape} respectively.'
        )

        self.recvbuf = np.zeros(shape, dtype=float)
        self.pq_sendbuf = np.zeros(pq_shape, dtype=float)
        self.pq_recvbuf = np.zeros(pq_shape, dtype=float)
        self.dpdq_sendbuf = np.zeros(dpdq_shape, dtype=float)
        self.dpdq_recvbuf = np.zeros(dpdq_shape, dtype=float)

        all_indicies = np.arange(ndets)
        proc_indicies = np.array_split(all_indicies, self.size)
        iproc_indicies = proc_indicies[self.rank]

        self.print(f'Generated vector indicies job map for size {ndets}:')
        self.print(f'{"iproc":>8} {"min(index)":>12} {"max(index)":>12}')

        for iproc in range(self.size):
            self.print(
                f'{iproc:>8} '
                f'{min(proc_indicies[iproc]):>12d} '
                f'{max(proc_indicies[iproc]):>12d} '
            )

            if iproc == self.rank:
                assert np.array_equal(iproc_indicies, proc_indicies[iproc])

        self.imin = min(iproc_indicies)
        self.imax = max(iproc_indicies)

        return

    def __repr__(self) -> None:
        r''' TODO: Docstring here.
        '''
        self.sleep()

        if self.parent:
            m = 'Hello from parent, '
        else:
            m = 'Hello from child, '

        m += f'I am rank {self.rank + 1} of {self.size} total cpu(s).'

        return m
