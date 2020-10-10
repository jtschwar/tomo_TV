CXX = g++ -fPIC -fopenmp
MPXX = mpicxx -fPIC -fopenmp
CUDAXX = nvcc -shared -Xcompiler -fPIC -c
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11  -I$(HOME)/.local/include/python3.6m/ `python3 -m pybind11 --includes`
EIGEN = -I /home/hlab/tomo_TV/eigen3 
ASTRA = -DASTRA_CUDA -I /home/hlab/astra-toolbox -I /home/hlab/astra-toolbox/include
ASTRA_LIB = -L /home/hlab/astra/lib/ -lastra
CUDA = -I /usr/lib/cuda/include -L /usr/lib/cuda/lib64 -lcudart -lz
HDF5_INC = -I /home/hlab/tomo_TV/HDF5/include
HDF5_LIBS= -L /home/hlab/tomo_TV/HDF5/lib -lhdf5 -lhdf5_hl
