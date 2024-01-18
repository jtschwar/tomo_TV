#Makefile
#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`
MMGPUCONFIG = mm_astra`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_astra_ctvlib`python3-config --extension-suffix`
MMTVCONFIG = MM_tv`python3-config --extension-suffix`

all: shared_library mm_astra 

shared_library:
	cd container; make; cd ..
	cd regularizers; make; cd ..
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

mm_astra: mm_astra.cpp mm_astra.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(NonPAR_HDF5_INC) mm_astra.cpp -o $(MMGPUCONFIG) $(ASTRA_LIB) aux_func.so 

clean:
	rm -rf *.so *.o


