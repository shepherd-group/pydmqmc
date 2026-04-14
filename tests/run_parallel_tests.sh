#!/bin/bash
# If your version of MPI does not support running mpi-pytest in forking mode
# (for instance, OpenMPI on Ubuntu), you can run the parallel tests using this script.

for i in {1..2}; do
    mpiexec -n $i pytest -m parallel[$i] -k _parallel
done