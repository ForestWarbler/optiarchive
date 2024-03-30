#!/bin/bash

mpirun -np 2 -host scc-1,scc-2 -ppn 1 ./mygemm -c case.dat -o result.dat
