#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_astra_ctvlib`python3-config --extension-suffix`

all: shared_library astra_ctvlib

shared_library:
	cd container; make; cd ..
	cd regularizers; make; cd ..
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

astra_ctvlib: astra_ctvlib.cpp astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) astra_ctvlib.cpp container/*.o regularizers/*.o -o $(GPUCONFIG) \
	$(ASTRA_LIB) \
	-Wl,-rpath,'$$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

clean:
	rm -rf *.so *.o