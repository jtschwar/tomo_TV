#!/bin/sh
include ./make.inc
#CXX = g++-8
#MPXX = mpic++
MPXX=CC
#CXXFLAGS = -O3 -Wno-div-by-zero -fopenmp -shared -std=c++11 -undefined dynamic_lookup -ffast-math -march=native
#EIGEN = -I /opt/local/include/eigen3 
#PYBIND11 = `python3 -m pybind11 --includes` 
PYCONFIG = ctvlib`python3-config --extension-suffix`
MPICONFIG = mpi_ctvlib`python3-config --extension-suffix`

ctvlib: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND) $(PYBIND11) ctvlib.cpp -o $(PYCONFIG) 

mpi_ctvlib: mpi_ctvlib.cpp mpi_ctvlib.hpp
	$(MPXX) $(CXXFLAGS) $(EIGEN) $(PYBIND) $(PYBIND11) mpi_ctvlib.cpp -o $(MPICONFIG) 
