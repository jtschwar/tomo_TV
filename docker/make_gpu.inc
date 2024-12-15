CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11 `python3 -m pybind11 --includes`
EIGEN = -I /tomo_TV/thirdparty/eigen
ASTRA = -DASTRA_CUDA -I /tomo_TV/thirdparty/astra-toolbox -I /tomo_TV/thirdparty/astra-toolbox/include
ASTRA_LIB = -L /tomo_TV/thirdparty/astra-toolbox/lib/ -lastra
CUDA = -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lz
HDF5_INC = -I /path/to/HDF5/include
HDF5_LIBS= -L /path/to/HDF5/lib -lhdf5 -lhdf5_hl
