#!/bin/sh
include ./make.inc

GPUCONFIG = tomoengine`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_tomoengine`python3-config --extension-suffix`

all: shared_library tomoengine

shared_library:
	cd container; make; cd ..
	cd regularizers; make; cd ..
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

tomoengine: tomoengine.cpp tomoengine.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) tomoengine.cpp container/*.o regularizers/*.o -o $(GPUCONFIG) \
	$(ASTRA_LIB) \
	-Wl,-rpath,'$$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

clean:
	rm -rf *.so *.o