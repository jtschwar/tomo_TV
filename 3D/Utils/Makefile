CXX = g++-8
CXXFLAGS = -O3 -Wno-div-by-zero -fopenmp -shared -std=c++11 -undefined dynamic_lookup -ffast-math -march=native
EIGEN = -I /opt/local/include/eigen3 
PYBIND11 = `python3 -m pybind11 --includes` 
ctvlib_config = ctvlib`python3-config --extension-suffix`

all: ctvlib

ctvlib: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(ctvlib_config) 
