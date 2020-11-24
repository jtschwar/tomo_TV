CXX = g++-8 -fPIC -fopenmp
MPXX = mpicc
CXXFLAGS = -O3 -Wno-div-by-zero -shared -std=c++11   -undefined dynamic_lookup -ffast-math -march=native
EIGEN = -I /opt/local/include/eigen3 
FFTW = -I /opt/local/include/
PYBIND11 = `python3 -m pybind11 --includes` 
HDF5_INC = -I /home/hlab/tomo_TV/HDF5/include
HDF5_LIBS= -L /home/hlab/tomo_TV/HDF5/lib -lhdf5 -lhdf5_hl

