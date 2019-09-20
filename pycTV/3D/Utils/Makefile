CXX = /opt/local/bin/g++-mp-8
CXXFLAGS = -O3 -Wno-div-by-zero -fopenmp -shared -std=c++11 -undefined dynamic_lookup -ffast-math -march=native
EIGEN = -I /opt/local/include/eigen3 
PYBIND11 = `python3 -m pybind11 --includes` 
PYCONFIG = ctvlib`python3-config --extension-suffix`

add: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(PYCONFIG) 
