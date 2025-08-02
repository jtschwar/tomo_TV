#Makefile
#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_astra_ctvlib`python3-config --extension-suffix`

all: shared_library astra_ctvlib mpi_astra_ctvlib  

shared_library:
	cd container; make; cd ..
	cd regularizers; make; cd .. 
	g++ -shared container/*.o regularizers/*.o -o aux_func.so

astra_ctvlib: astra_ctvlib.cpp astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(PYBIND) $(PYBIND11) astra_ctvlib.cpp -o $(GPUCONFIG) $(ASTRA_LIB) aux_func.so 

mpi_astra_ctvlib: mpi_astra_ctvlib.cpp mpi_astra_ctvlib.hpp
	$(MPXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(HDF5_INC) $(PYBIND) $(PYBIND11) mpi_astra_ctvlib.cpp -o $(MPIGPUCONFIG) $(ASTRA_LIB) $(HDF5_LIBS) aux_func.so

clean:
	rm -rf *.so *.o
