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
        parallel_assert(self._ph.comm is not None, msg=f"rank {self._ph.rank}: comm should not be None")
        parallel_assert(isinstance(self._ph.rank, int), msg=f"rank {self._ph.rank}: rank should be int")
        parallel_assert(isinstance(self._ph.size, int), msg=f"rank {self._ph.rank}: size should be int")
        parallel_assert(isinstance(self._ph.root, int), msg=f"rank {self._ph.rank}: root should be int")
        parallel_assert(isinstance(self._ph.parent, bool), msg=f"rank {self._ph.rank}: parent should be bool")
        parallel_assert(isinstance(self._ph.parallel, bool), msg=f"rank {self._ph.rank}: parallel should be bool")
        parallel_assert(isinstance(self._ph.imin, int), msg=f"rank {self._ph.rank}: imin should be int")
        parallel_assert(isinstance(self._ph.imax, int), msg=f"rank {self._ph.rank}: imax should be int")
        parallel_assert(self._ph.bufsize == self._matrix_size, msg=f"rank {self._ph.rank}: bufsize expected {self._matrix_size}, got {self._ph.bufsize}")
        parallel_assert(self._ph.bufshape == self._matrix_shape, msg=f"rank {self._ph.rank}: bufshape expected {self._matrix_shape}, got {self._ph.bufshape}")

    @mark.parallel(1)
    def test_setup_job_map_1proc(self):
        answers = {0: (0, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0],
                        msg=f"rank {self._ph.rank}: imin expected {answers[self._ph.rank][0]}, got {self._ph.imin}")
        parallel_assert(self._ph.imax == answers[self._ph.rank][1],
                        msg=f"rank {self._ph.rank}: imax expected {answers[self._ph.rank][1]}, got {self._ph.imax}")

    @mark.parallel(2)
    def test_setup_job_map_2proc(self):
        answers = {0: (0, 5), 1: (5, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0],
                        msg=f"rank {self._ph.rank}: imin expected {answers[self._ph.rank][0]}, got {self._ph.imin}")
        parallel_assert(self._ph.imax == answers[self._ph.rank][1],
                        msg=f"rank {self._ph.rank}: imax expected {answers[self._ph.rank][1]}, got {self._ph.imax}")

    @mark.parallel(3)
    def test_setup_job_map_3proc(self):
        answers = {0: (0, 4), 1: (4, 7), 2: (7, 10)}
        parallel_assert(self._ph.imin == answers[self._ph.rank][0],
                        msg=f"rank {self._ph.rank}: imin expected {answers[self._ph.rank][0]}, got {self._ph.imin}")
        parallel_assert(self._ph.imax == answers[self._ph.rank][1],
                        msg=f"rank {self._ph.rank}: imax expected {answers[self._ph.rank][1]}, got {self._ph.imax}")

    @mark.parallel([2,3])
    def test_setup_job_map_oversubscribe(self):
        with raises(ValueError):
            ParallelHelper(shape=(1, 10))

    @mark.parallel(1)
    def test_get_rng_seed_1proc(self):
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == 100, msg=f"rank {self._ph.rank}: seed expected 100, got {seed}")

    @mark.parallel(2)
    def test_get_rng_seed_2proc(self):
        seeds = [100, 102]
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == seeds[self._ph.rank],
                        msg=f"rank {self._ph.rank}: seed expected {seeds[self._ph.rank]}, got {seed}")

    @mark.parallel(3)
    def test_get_rng_seed_3proc(self):
        seeds = [100, 103, 106]
        seed = self._ph.get_rng_seed(100)
        parallel_assert(seed == seeds[self._ph.rank],
                        msg=f"rank {self._ph.rank}: seed expected {seeds[self._ph.rank]}, got {seed}")

    @mark.parallel([1,2,3])
    def test_allocate_buffers(self):
        self._ph.allocate_reduce_buffers()
        parallel_assert(self._ph._recvbuf is not None,
                        msg=f"rank {self._ph.rank}: _recvbuf should not be None after allocation")
        parallel_assert(self._ph._recvbuf.shape == self._matrix_shape,
                        msg=f"rank {self._ph.rank}: recvbuf shape expected {self._matrix_shape}, got {self._ph._recvbuf.shape}")

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
        parallel_assert(np.array_equal(array, answer),
                        msg=f"rank {self._ph.rank}: broadcast array expected {answer}, got {array}")

    @mark.parallel([1,2,3])
    def test_allreduce_sum_without_allocation(self):
        array = np.zeros(self._matrix_shape)
        with raises(RuntimeError):
            self._ph.allreduce_sum(array)

    @mark.parallel([1,2,3])
    def test_allreduce_sum_incorrect_size(self):
        self._ph.allocate_reduce_buffers()
        array = np.zeros((self._matrix_shape[0], self._matrix_shape[1] + 1))
        with raises(ValueError):
            self._ph.allreduce_sum(array)

    @mark.parallel([1,2,3])
    def test_allreduce_sum(self):
        local_array = np.ones(self._matrix_shape)
        answer = local_array * self._ph.size
        
        self._ph.allocate_reduce_buffers()
        
        res = self._ph.allreduce_sum(local_array)
        
        parallel_assert(np.array_equal(res, answer),
                        msg=f"rank {self._ph.rank}: allreduce sum expected {answer}, got {res}")
