# Building tomo_TV

This project serves as a wrapper around [ASTRA-toolbox](https://github.com/astra-toolbox/astra-toolbox) for GPU acccelerated and custom written parallel C++ reconstraction algorithms. In both cases, we recommend using the GNU GCC compiler, any version above 4.8 that supports [OpenMP](https://gcc.gnu.org/wiki/openmp) is suitable for compiling these scripts. To install the C++ accelerated scripts:

    cd cpu_3D/Utils/
    make ctvlib

On MacOS: the build command requires passing of the `-undefined dynamic_lookup` flag.  If you're compiling the C++ for any other operating system remove that flag from line. M1 and any future architectures requires the passing of an additional  `-mcpu=apple-m1` flag. 

To compile multi - nodal scripts compile the mpi_ctvlib instead (assuming MPI is available).

# Compiling GPU Accelerated Reconstruction Scripts

For GPU accelerated forward and back-projection operators, we first will need to be in the ASTRA-toolbox folder in `thirdparty/astra-toolbox`. Tomo_tv uses a couple custom written C++ scripts to link the two packages together. Complete the following steps to build Astra from source:

    cd thirdparty/astra-toolbox/build/linux
    ./autogen.sh
    .configure --with-cuda=/path/to/cuda --with-python --with-install-type=prefix --prefix=/path/to/astra
    make all

To get your cuda path run: `whereis cuda`. You'll also need to specify the path where you would like to build the Astra libary. I'd recommend spefcifying the path to where tomo_TV is located. Example: `/path/to/tomo_TV/astra`.

After the configuration compeletes, add `--prefix=/path/to/astra` to the end of L507 of the Makefile and finish installing the package.

    make install # this will install the Python package to /path/to/astra

Once Astra is available, now let's compile tomo_TV. 

    cd tomo_TV/gpu_3D/Utils
    make shared library

Before compiling astra_ctvlib we'll need to specify the paths for ASTRA and CUDA. Open make.inc and edit the following lines 

    ASTRA_LIB = -L /path/to/astra/lib -lasatra
    CUDA = -I /path/to/cuda/include -L /path/to/cuda/lib64 -lcudart -lz

Complete the process by running: 

    make astra_ctvlib

# Compiling Multi-GPU or Mutli-Node (HPC) Support

Multi-GPU parallel scripts requires parallel HDF5 as an additional dependency. 

To run the scripts enter : `mpirun -n XX python3 dynamicTomo.py` where XX is the number of cores or GPUs available. \\

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations (available as a thirdparty submodule).  vc