#!/bin/bash

mpirun -np 2 -host scc-1,scc-2 -ppn 1 ./myjacobi -c case.dat -o result.dat
