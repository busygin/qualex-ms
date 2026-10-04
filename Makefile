# Makefile of QUALEX-MS solver for GNU make
#
#   make              the solver on the CPU: LAPACK and CBLAS from OpenBLAS
#   make BLAS=mkl     the same on Intel MKL instead of OpenBLAS
#   make GPU=1        the solver on cuSOLVER and cuBLAS (needs the CUDA toolkit),
#                     the Douglas-Rachford stage still taking CBLAS from BLAS
#
# MKL comes from MKLROOT, by default ~/opt/intel-mkl-2026.1, where the libraries
# of the PyPI wheel mkl are unpacked on the development machine; for an oneAPI
# installation set MKLROOT, or MKL_LIBS to its whole link line.
#
# The library (lib/, see lib/Makefile) and the solver are built in
# build/<cpu|gpu>-<openblas|mkl>, and ./qualex-ms is a copy of the solver of the
# configuration asked for.

GPU ?= 0
BLAS ?= openblas
BACKEND := $(if $(filter 1,$(GPU)),gpu,cpu)
BUILD := build/$(BACKEND)-$(BLAS)

MKLROOT ?= $(HOME)/opt/intel-mkl-2026.1
MKL_LIB = $(MKLROOT)/lib
# the GNU threading layer; the old-style rpath also covers the CPU-specific
# kernels MKL opens at run time
MKL_LIBS ?= $(MKL_LIB)/libmkl_gf_lp64.so.3 $(MKL_LIB)/libmkl_gnu_thread.so.3 \
  $(MKL_LIB)/libmkl_core.so.3 -lgomp -lpthread -lm -ldl \
  -Wl,--disable-new-dtags,-rpath,$(MKL_LIB)

ifeq ($(BLAS),openblas)
BLAS_LIBS = -lopenblas
else ifeq ($(BLAS),mkl)
BLAS_LIBS = $(MKL_LIBS)
else
$(error BLAS must be openblas or mkl)
endif
LIBS = $(BLAS_LIBS) $(if $(filter 1,$(GPU)),-lcudart -lcusolver -lcublas)

CXX = g++
CXXFLAGS = -std=gnu++20 -DNDEBUG -Ofast -funroll-all-loops -Wall -Ilib
LINKFLAGS = -s

qualex-ms: $(BUILD)/qualex-ms FORCE
	@cmp -s $< $@ || cp $< $@

$(BUILD)/qualex-ms: $(BUILD)/main.o $(BUILD)/libqms.a
	$(CXX) $(LINKFLAGS) -o $@ $^ $(LIBS)

$(BUILD)/libqms.a: $(wildcard lib/*.cc lib/*.c lib/*.h lib/Makefile)
	$(MAKE) -C lib GPU=$(GPU) BUILD=$(abspath $(BUILD))

$(BUILD)/main.o: main.cc $(wildcard lib/*.h)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build qualex-ms

FORCE:

.PHONY: clean FORCE
