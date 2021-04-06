#Makefile
#!/bin/sh
include ./make.inc

PYCONFIG = ctvlib`python3-config --extension-suffix`

all: ctvlib
ctvlib: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(TBB) $(TBBLIB) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(PYCONFIG)

clean:
	rm -rf *.so *.o
