# Makefile of the QUALEX-MS library, libqms.a, for GNU make
#
#   make [GPU=1|0] [BUILD=<dir>]       libqms.a, the whole library
#   make core [BUILD=<dir>]            libqms_core.a, the combinatorial core
#
# GPU=1 (the default) builds the linear algebra backend on cuSOLVER and cuBLAS,
# GPU=0 the one on LAPACK and CBLAS.  Objects and the libraries go to BUILD
# (default build/gpu or build/cpu in this directory).  A program linking
# libqms.a needs
#   -lopenblas -lcudart -lcusolver -lcublas   with GPU=1
#   -lopenblas                                with GPU=0
# (OpenBLAS supplies CBLAS, which the Douglas-Rachford stage uses on either
# backend, and LAPACK).  libqms_core.a (bit vectors, graphs, the greedy
# heuristics with MIN, preprocessing) needs nothing beyond the C++ library.

GPU ?= 1
BACKEND := $(if $(filter 1,$(GPU)),gpu,cpu)
BUILD ?= build/$(BACKEND)

CC = gcc
CXX = g++
CFLAGS = -DNDEBUG -Ofast -funroll-all-loops -Wall
CXXFLAGS = -std=gnu++20 $(CFLAGS)

# the combinatorial core, which needs no BLAS
CORE = bool_vector.cc graph.cc greedy_clique.cc refiner.cc preproc_clique.cc
# the spectral stages
SPECTRAL = wrapper.cc qualex.cc dr.cc

CORE_OBJS = $(addprefix $(BUILD)/,$(CORE:.cc=.o))
OBJS = $(CORE_OBJS) $(addprefix $(BUILD)/,$(SPECTRAL:.cc=.o) linalg_$(BACKEND).o)
HEADERS = $(wildcard *.h)

$(BUILD)/libqms.a: $(OBJS)
	rm -f $@
	ar rcs $@ $^

core: $(BUILD)/libqms_core.a

$(BUILD)/libqms_core.a: $(CORE_OBJS)
	rm -f $@
	ar rcs $@ $^

$(BUILD)/%.o: %.cc $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: %.c linalg.h | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

clean:
	rm -rf build

.PHONY: core clean
