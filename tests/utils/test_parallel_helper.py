import numpy as np
from pytest import fixture, raises, mark
from pytest_mpi import parallel_assert

from pydmqmc.utils.parallel_helper import ParallelHelper


class TestParallelHelper_Parallel():
    """Test the ParallelHelper class."""

    @fixture(autouse=True)
    def _setup(self):
        self._matrix_shape = (10, 10)
        self._matrix_size = np.prod(self._matrix_shape)
        self._ph = ParallelHelper(self._matrix_shape)

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
        parallel_assert(self._ph.bufsize == self._matrix_size)
        parallel_assert(self._ph.bufshape == self._matrix_shape)

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
        parallel_assert(self._ph._recvbuf.shape == self._matrix_shape)

    @mark.parallel([1,2,3])
    def test_broadcast(self):
        answer = np.arange(self._matrix_size, dtype=float)
        answer = answer.reshape(self._matrix_shape)
        
        if self._ph.parent:
            array = np.arange(self._matrix_size, dtype=float)
            array = array.reshape(self._matrix_shape)
        else:
            array = np.empty(self._matrix_shape, dtype=float)
        
        self._ph.broadcast(array)
        parallel_assert(np.array_equal(array, answer))

    @mark.parallel([1,2,3])
    def test_allreduce_sum_without_allocation(self):
        array = np.zeros(self._matrix_shape)
        with raises(RuntimeError):
            self._ph.allreduce_sum(array)

    @mark.parallel([1,2,3])
    def test_allreduce_sum_incorrect_size(self):
        self._ph.allocate_buffers()
        array = np.zeros((self._matrix_shape[0], self._matrix_shape[1] + 1))
        with raises(ValueError):
            self._ph.allreduce_sum(array)

    @mark.parallel([1,2,3])
    def test_allreduce_sum(self):
        local_array = np.ones(self._matrix_shape)
        answer = local_array * self._ph.size
        
        self._ph.allocate_buffers()
        
        res = self._ph.allreduce_sum(local_array)
        
        parallel_assert(np.array_equal(res, answer))
