CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11  `python3 -m pybind11 --includes`
EIGEN = -I ../../../thirdparty/eigen 
ASTRA = -DASTRA_CUDA -I ../../../thirdparty/astra-toolbox -I ../../../thirdparty/astra-toolbox/include
ASTRA_LIB = -L ../../../thirdparty/astra-toolbox/lib/ -lastra
CUDA = -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lz
