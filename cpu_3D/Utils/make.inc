CXX = g++ -fPIC -Xclang -fopenmp
MPXX = mpicc
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11 -ffast-math 
EIGEN = -I ../../thirdparty/eigen 
PYBIND11 = `python3 -m pybind11 --includes` 

