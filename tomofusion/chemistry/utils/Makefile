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
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) multimodal.cpp container/*.o regularizers/*.o -o $(MMGPUCONFIG) \
	$(ASTRA_LIB) \
	-Wl,-rpath,'$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

multigpufusion: multigpufusion.cpp multigpufusion.hpp multimodal.cpp multimodal.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) multigpufusion.cpp multimodal.cpp container/*.o regularizers/*.o -o $(MULTIMMGPUCONFIG) \
	$(ASTRA_LIB) \
	-Wl,-rpath,'$ORIGIN' \
	-Wl,-rpath,$(shell pwd)

clean:
	cd container && make clean && cd ..
	cd regularizers && make clean && cd ..
	rm -rf *.so *.o