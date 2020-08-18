#Makefile
#!/bin/sh
include ./make.inc

GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`
MPIGPUCONFIG = mpi_astra_ctvlib`python3-config --extension-suffix`


all: mpi_astra_ctvlib astra_ctvlib

tv_gd: regularizers/tv_gd.cu regularizers/tv_gd.h
	$(CUDAXX) regularizers/tv_gd.cu 

astra_ctvlib: astra_ctvlib.cpp astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(PYBIND) $(PYBIND11) astra_ctvlib.cpp -o $(GPUCONFIG) $(ASTRA_LIB) container/Matrix3D.o container/matrix_ops.o tv_gd.o 

mpi_astra_ctvlib: mpi_astra_ctvlib.cpp mpi_astra_ctvlib.hpp
	$(MPXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(PYBIND) $(PYBIND11) mpi_astra_ctvlib.cpp -o $(MPIGPUCONFIG) $(ASTRA_LIB) 

clean:
	rm -rf *.so *.o
