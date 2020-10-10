CXX = g++-8 -fPIC -fopenmp
MPXX = mpicc
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11 `python3 -m pybind11 --includes`  -undefined dynamic_lookup -ffast-math -march=native
EIGEN = -I /opt/local/include/eigen3 
FFTW = -I /opt/local/include/
PYBIND11 = `python3 -m pybind11 --includes` 
