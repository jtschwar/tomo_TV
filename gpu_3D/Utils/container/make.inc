CXX = g++ -fPIC -shared 
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -std=c++11 -I/home/hlab/tomo_TV/eigen3
