CC ?= cc
MPICC ?= mpicc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
OMPFLAGS ?= -fopenmp
OMPLIBS ?=

REFERENCE = ../src/reference/graph.c ../src/reference/features.c

all: openmp_graphfeat mpi_graphfeat speedupplots

openmp_graphfeat: OpenMP.c $(REFERENCE) ../src/reference/graph.h ../src/reference/features.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(OMPFLAGS) OpenMP.c $(REFERENCE) $(OMPLIBS) -lm -o $@

mpi_graphfeat: MPI.c $(REFERENCE) ../src/reference/graph.h ../src/reference/features.h
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $(OMPFLAGS) MPI.c $(REFERENCE) $(OMPLIBS) -lm -o $@

speedupplots: speedupplots.c
	$(CC) $(CFLAGS) speedupplots.c -o $@

clean:
	rm -f openmp_graphfeat mpi_graphfeat speedupplots

.PHONY: all clean
