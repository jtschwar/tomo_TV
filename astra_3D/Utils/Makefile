#Makefile
#!/bin/sh
include ./make.inc

CPUCONFIG = ctvlib`python3-config --extension-suffix`
MPICONFIG = mpi_ctvlib`python3-config --extension-suffix`
GPUCONFIG = astra_ctvlib`python3-config --extension-suffix`

%.o : %.C
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) -c $< -o $@

all: ctvlib mpi_ctvlib astra_ctvlib

ctvlib: ctvlib.o  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND) $(PYBIND11) ctvlib.o -o $(CPUCONFIG)

mpi_ctvlib: mpi_ctvlib.o mpi_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND) $(PYBIND11) mpi_ctvlib.o -o $(MPICONFIG)

astra_ctvlib: astra_ctvlib.o astra_ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(ASTRA) $(CUDA) $(PYBIND) $(PYBIND11) astra_ctvlib.o -o $(GPUCONFIG) $(ASTRA_LIB) 

clean:
	rm -rf *.so *.o
