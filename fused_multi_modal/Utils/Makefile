#Makefile
#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`
MMGPUCONFIG = mm_astra`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_astra_ctvlib`python3-config --extension-suffix`
MMTVCONFIG = MM_tv`python3-config --extension-suffix`

all: shared_library astra_ctvlib mpi_astra_ctvlib mm_astra MM_tv

shared_library: 
	nvcc -shared container/*.o regularizers/*.o -o aux_func.so

astra_ctvlib: astra_ctvlib.cpp astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) astra_ctvlib.cpp -o $(GPUCONFIG) $(ASTRA_LIB) aux_func.so

mpi_astra_ctvlib: mpi_astra_ctvlib.cpp mpi_astra_ctvlib.hpp
	$(MPXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(HDF5_INC) mpi_astra_ctvlib.cpp -o $(MPIGPUCONFIG) $(ASTRA_LIB) $(HDF5_LIBS) aux_func.so

MM_tv: MM_tv.cpp MM_tv.hpp
	g++ -fPIC -Wno-div-by-zero -shared -std=c++11  -O3 -march=native  -I ../../thirdparty/eigen `python3 -m pybind11 --includes`  MM_tv.cpp -o MM_tv`python3-config --extension-suffix`

mm_astra: mm_astra.cpp mm_astra.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(NonPAR_HDF5_INC) mm_astra.cpp -o $(MMGPUCONFIG) $(ASTRA_LIB) aux_func.so 

clean:
	rm -rf *.so *.o


