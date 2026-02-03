import numpy as np
from pytest import fixture, raises, mark
from pytest_mpi import parallel_assert

from pydmqmc.utils.parallel_helper import ParallelHelper


class TestParallelHelper():
    """Test the ParallelHelper class."""

    @fixture(autouse=True)
    def _setup(self):
        self._vector_size = 10
        self._initiator_threshold = 0
        self._ph = ParallelHelper(self._vector_size, self._initiator_threshold)

    @mark.parallel([1,2,3])
    def test_properties(self):
        parallel_assert(self._ph.comm is not None)
        parallel_assert(isinstance(self._ph.rank, int))
        parallel_assert(isinstance(self._ph.size, int))
        parallel_assert(isinstance(self._ph.root, int))
        parallel_assert(isinstance(self._ph.parent, bool))
        parallel_assert(isinstance(self._ph.parallel, bool))
        parallel_assert(isinstance(self._ph.imin, int))
        parallel_assert(isinstance(self._ph.imax, int))
        parallel_assert(self._ph.bufshape == (1, self._vector_size))

    def test_nonzero_initiator_threshold(self):
        ph = ParallelHelper(self._vector_size, initiator_threshold=5)
        parallel_assert(ph.bufshape == (5, self._vector_size))

    @mark.parallel(1)
    def test_setup_job_map_1proc(self):
        answers = {0: (0, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0])
        parallel_assert(self._ph.imax == answers[self._ph.rank][1])

    @mark.parallel(2)
    def test_setup_job_map_2proc(self):
        answers = {0: (0, 5), 1: (5, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0])
        parallel_assert(self._ph.imax == answers[self._ph.rank][1])

    @mark.parallel(3)
    def test_setup_job_map_3proc(self):
        answers = {0: (0, 4), 1: (4, 7), 2: (7, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0])
        parallel_assert(self._ph.imax == answers[self._ph.rank][1])

    @mark.parallel(1)
    def test_get_rng_seed_1proc(self):
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == 100)

    @mark.parallel(2)
    def test_get_rng_seed_2proc(self):
        seeds = [100, 102]
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == seeds[self._ph.rank])

    @mark.parallel(3)
    def test_get_rng_seed_3proc(self):
        seeds = [100, 103, 106]
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == seeds[self._ph.rank])

    @mark.parallel([1,2,3])
    def test_allocate_buffers(self):
        self._ph.allocate_buffers()
        parallel_assert(self._ph._recvbuf is not None)
        parallel_assert(self._ph._recvbuf.size == self._vector_size)
        parallel_assert(self._ph._recvbuf.shape == (1, self._vector_size))

    @mark.parallel([1,2,3])
    def test_broadcast(self):
        answer = np.arange(self._vector_size, dtype=float)
        
        if self._ph.parent:
            array = np.arange(self._vector_size, dtype=float)
        else:
            array = np.empty(self._vector_size, dtype=float)
        
        self._ph.bcast(array)
        parallel_assert(np.array_equal(array, answer))

    @mark.parallel([1,2,3])
    def test_sum_reduce_without_allocation(self):
        array = np.zeros(self._vector_size)
        with raises(RuntimeError):
            self._ph.sum_reduce(array)

    @mark.parallel([1,2,3])
    def test_sum_reduce_incorrect_size(self):
        self._ph.allocate_buffers()
        array = np.zeros((self._vector_size + 1,))
        with raises(ValueError):
            self._ph.sum_reduce(array)

    @mark.parallel([1,2,3])
    def test_sum_reduce(self):
        local_array = np.ones((1, self._vector_size))
        answer = local_array * self._ph.size
        
        self._ph.allocate_buffers()
        
        res = self._ph.sum_reduce(local_array)
        
        if self._ph.parent:
            parallel_assert(np.array_equal(res, answer))
        else:
            parallel_assert(np.array_equal(res, local_array))