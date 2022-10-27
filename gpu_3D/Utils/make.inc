CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11 `python3 -m pybind11 --includes`
EIGEN = -I ../../thirdparty/eigen 
ASTRA = -DASTRA_CUDA -I ../../thirdparty/astra-toolbox -I ../../thirdparty/astra-toolbox/include
ASTRA_LIB = -L /path/to/astra/lib/ -lastra
CUDA = -I /path/to/cuda/include -L /path/to/cuda/lib64 -lcudart -lz
HDF5_INC = -I /path/to/HDF5/include
HDF5_LIBS= -L /path/to/HDF5/lib -lhdf5 -lhdf5_hl
