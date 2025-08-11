#!/bin/sh
include ./make.inc

MMGPUCONFIG = multimodal`python3-config --extension-suffix`
MULTIMMGPUCONFIG = multigpufusion`python3-config --extension-suffix`

all: shared_library multimodal multigpufusion

shared_library:
	cd container; make; cd ..
	cd regularizers; make; cd ..
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

multimodal: multimodal.cpp multimodal.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(NonPAR_HDF5_INC) multimodal.cpp -o $(MMGPUCONFIG) \
	$(ASTRA_LIB) aux_func.so \
	-Wl,-rpath,'$$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

multigpufusion: multigpufusion.cpp multigpufusion.hpp multimodal.cpp multimodal.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) multigpufusion.cpp multimodal.cpp container/*.o regularizers/*.o -o $(MULTIGPUFUSIONCONFIG) \
	$(ASTRA_LIB) \
	-Wl,-rpath,'$$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

clean:
	rm -rf *.so *.o
