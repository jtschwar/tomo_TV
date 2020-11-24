#Makefile
#!/bin/sh
include ./make.inc

PYCONFIG = ctvlib`python3-config --extension-suffix`
MPICONFIG = mpi_ctvlib`python3-config --extension-suffix`
all: ctvlib mpi_ctvlib
ctvlib: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(PYCONFIG)

mpi_ctvlib: mpi_ctvlib.cpp mpi_ctvlib.hpp
	$(MPXX) $(CXXFLAGS) $(EIGEN) $(HDF5_INC) $(PYBIND11) mpi_ctvlib.cpp -o $(MPICONFIG) $(HDF5_LIBS)
clean:
	rm -rf *.so *.o
