CC = icc

CFLAGS	= -std=c++11 -O3 -ipo -fp-model fast=2 -xHost -qopt-matmul -qopt-zmm-usage=high -qopenmp

all: mycov fpdiff

mycov: covariance.o
	$(CC) -o $@ $^ $(CFLAGS)

covariance.o: covariance.cpp
	$(CC) -c $(CFLAGS) $<

fpdiff: fpdiff.cpp
	$(CC) -o $@ $^

gencase: gencase.cpp
	$(CC) -o $@ $^

.PHONY: clean

clean: 
	rm -f *.o
	rm -f mycov
	rm -f fpdiff
	rm -f gencase
