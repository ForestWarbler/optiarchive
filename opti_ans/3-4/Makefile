CC = icc

CFLAGS	= -O3 -qopenmp

all: myseidel fpdiff

myseidel: seidel-2d.o
	$(CC) -o $@ $^ $(CFLAGS)

seidel-2d.o: seidel-2d.cpp
	$(CC) -c $(CFLAGS) $<

fpdiff: fpdiff.cpp
	$(CC) -o $@ $^

gencase: gencase.cpp
	$(CC) -o $@ $^

.PHONY: clean

clean: 
	rm -f *.o
	rm -f myseidel
	rm -f fpdiff
	rm -f gencase
