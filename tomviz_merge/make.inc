CXX = g++-8 -fPIC -fopenmp
MPXX = mpicc
CXXFLAGS = -O3 -Wno-div-by-zero -shared -undefined dynamic_lookup -ffast-math -march=native -DNDEBUG
EIGEN = -I /opt/local/include/eigen3 
PYBIND11 = `python3 -m pybind11 --includes` 
TBB = -I /usr/local/Cellar/tbb/2019_U5_1/include
TBBLIB = -L /usr/local/Cellar/tbb/2019_U5_1/lib -ltbb