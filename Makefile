# Makefile of QUALEX-MS solver for GNU make
#
#   make          the solver on the cuSOLVER/cuBLAS backend: needs the CUDA
#                 toolkit (cuSOLVER, cuBLAS, cudart) and OpenBLAS
#   make GPU=0    the solver on the LAPACK/CBLAS backend: needs OpenBLAS only
#
# The library (lib/, see lib/Makefile) and the solver are built in build/gpu or
# build/cpu, and ./qualex-ms is a copy of the solver of the backend asked for.

GPU ?= 1
BACKEND := $(if $(filter 1,$(GPU)),gpu,cpu)
BUILD := build/$(BACKEND)
LIBS := -lopenblas $(if $(filter 1,$(GPU)),-lcudart -lcusolver -lcublas)

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
