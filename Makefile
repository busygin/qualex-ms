# Makefile of QUALEX-MS solver for GNU make

# Requires: CUDA toolkit (cuSOLVER, cuBLAS, cudart)

CFLAGS = -DNDEBUG -Ofast -funroll-all-loops -s -Wall
LINKFLAGS = -s

CC = gcc
CXX = g++
LIBS = -lcudart -lcusolver -lcublas

.SUFFIXES: .o .cc .c

OBJS_c = bool_vector.cc graph.cc greedy_clique.cc main.cc preproc_clique.cc qualex.cc refiner.cc eigen.c mdv.c
OBJS_o = bool_vector.o graph.o greedy_clique.o main.o preproc_clique.o qualex.o refiner.o eigen.o mdv.o

qualex-ms: $(OBJS_c) $(OBJS_o)
	$(CXX) $(LINKFLAGS) $(CFLAGS) -o $@ $(OBJS_o) $(LIBS)

clean:
	rm *.o qualex-ms

.c.o:
	$(CC) $(CFLAGS) -c $<

.cc.o:
	$(CXX) $(CFLAGS) -c $<
