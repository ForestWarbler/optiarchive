CC = mpiicc

CXX = mpiicxx

CFLAGS	= -std=c++11 -O3 -ipo -fp-model fast=2 -xHost -qopt-matmul -qopt-zmm-usage=high -qopenmp

all: myjacobi fpdiff

myjacobi: jacobi-2d.o
	$(CC) -o $@ $^ $(CFLAGS)

jacobi-2d.o: jacobi-2d.cpp
	$(CC) -c $(CFLAGS) $<

fpdiff: fpdiff.cpp
	$(CC) -o $@ $^

gencase: gencase.cpp
	$(CC) -o $@ $^

.PHONY: clean

clean: 
	rm -f *.o
	rm -f myjacobi
	rm -f fpdiff
	rm -f gencase
