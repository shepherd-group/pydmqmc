# Disable Numba's JIT so that compiled functions are correctly included.
# Numba *says* it should do this automatically but experience counterindicates.
# Does make the test suite extremely slow though :c
export NUMBA_DISABLE_JIT=1

# Run coverage on the serial functionality.
# This takes the longest.
coverage run -m pytest

# Run coverage on the parallel functionality.
# We can re-enable Numba as JIT'ed functions aren't parallel exclusive.
# Some tests have only 1, 2, or 3 processes so we have to do it repeatedly.
export NUMBA_DISABLE_JIT=0
for i in {1..3}; do
    mpiexec -n $i coverage run -a -m pytest -m parallel[$i] -k _parallel
done