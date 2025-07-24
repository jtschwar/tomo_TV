CXX = g++
CXXFLAGS = -Wno-div-by-zero -shared -std=c++11 -undefined dynamic_lookup
CXXOPT = -O3 -march=native  
EIGEN = -I /opt/local/include/eigen3 
PYBIND11 = `python3 -m pybind11 --includes` 
PYCONFIG = ctvlib`python3-config --extension-suffix`

add: ctvlib.cpp  ctvlib.hpp
	$(CXX) $(CXXFLAGS) $(CXXOPT) $(EIGEN) $(PYBIND11) ctvlib.cpp -o $(PYCONFIG) 
