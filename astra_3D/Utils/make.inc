CXX = g++ -fPIC -fopenmp
MPXX = CC -qopenmp
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11  -I$(HOME)/.local/include/python3.6m/ `python3 -m pybind11 --includes`
EIGEN = -I ../../../eigen3/
ASTRA = -DASTRA_CUDA -I/ccs/home/jtschw/astra-toolbox -I/ccs/home/jtschw/astra-toolbox/include
ASTRA_LIB = -L/ccs/home/jtschw/astra/lib/ -lastra
CUDA = -I /sw/summit/cuda/10.1.243/include -L /sw/summit/cuda/10.1.243/lib64 -lz
