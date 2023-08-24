CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11  `python3 -m pybind11 --includes`
EIGEN = -I ../../thirdparty/eigen 
ASTRA = -DASTRA_CUDA -I /path/to/astra-toolbox -I /path/to/astra-toolbox/include
ASTRA_LIB = -L /path/to/astra/lib/ -lastra
CUDA = -I /path/to/cuda/include -L /path/to/cuda/lib64 -lcudart -lz
