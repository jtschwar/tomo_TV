# Building tomo_TV

This project serves as a wrapper around [ASTRA-toolbox](https://github.com/astra-toolbox/astra-toolbox) for GPU acccelerated and custom written parallel C++ reconstraction algorithms. In both cases, we recommend using the GNU GCC compiler, any version above 4.8 that supports [OpenMP](https://gcc.gnu.org/wiki/openmp) is suitable for compiling these scripts. To install the C++ accelerated scripts:

## Compiling C++ Accelerated Reconstruction Scripts

Compiling the C++ scripts is straightforward assuming GCC is avaiable and all the thirdparty dependencies are cloned. Simply move the Utility folder and compile the core script.

    cd cpu_3D/Utils/
    make ctvlib

On MacOS: the build command requires passing of the `-undefined dynamic_lookup` flag.  If you're compiling the C++ for any other operating system remove that flag from line. M1 and any future architectures requires the passing of an additional  `-mcpu=apple-m1` flag. You can append these flags in the `CXXFLAGS` variable in make.inc. 

Compiling ctvlib will create a shared object file in the following format: (ctvlib.cpython-3X-___.so). You can now import ctvlib as a python module and initialize it as a class. Refer to the demo jupyter-notebook for  more details. 

To compile multi - nodal scripts compile the mpi_ctvlib library instead (assuming MPI is available).

## Compiling GPU Accelerated Reconstruction Scripts

For GPU accelerated forward and back-projection operators, we first will need to be in the ASTRA-toolbox folder in `thirdparty/astra-toolbox`. Tomo_tv uses a couple custom written C++ scripts to link the two packages together.

Note: boost is a requirement for astra. Run `sudo apt-get install libboost-all-dev` to install boost or compile it from source (https://www.boost.org/doc/libs/1_43_0/more/getting_started/unix-variants.html).  

Complete the following steps to build Astra from source:

    cd thirdparty/astra-toolbox/build/linux
    ./autogen.sh
    ./configure --with-cuda=/path/to/cuda --with-python --with-install-type=prefix --prefix=/path/to/astra
    make all

To get your cuda path run: `whereis cuda`. You'll also need to specify the path where you would like to build the Astra libary. I'd recommend spefcifying the path to where tomo_TV is located. Example: `/path/to/tomo_TV/thirdparty/astra`.

After the configuration compeletes, add `--prefix=/path/to/astra` to the end of Line 508 in the Makefile where `/path/to/astra` is the path specified as the prefix in the above (e.g. `/path/to/tomo_TV/thirdparty/astra`). 

Finish installing the package.

    make install 

This will install the Python package to /path/to/astra (e.g. /path/to/tomo_TV/thirdparty/astra)

Once Astra is available, now let's compile tomo_TV. 

    cd tomo_TV/gpu_3D/Utils
    make shared_library

Before compiling astra_ctvlib we'll need to specify the paths for ASTRA and CUDA. Open make.inc and edit the following lines 

    ASTRA_LIB = -L /path/to/astra/lib -lasatra
    CUDA = -I /path/to/cuda/include -L /path/to/cuda/lib64 -lcudart -lz

Complete the process by running: 

    make astra_ctvlib

This will create a shared object file in the following format: (astra_ctvlib.cpython-3X-___.so). The final step requires linking the shared libraries and adding the necessary modules to the python path. Refer to `setup_tomo_tv.sh` for a sample shell script. 

## Compiling Multi-GPU or Mutli-Node (HPC) Support

Multi-GPU parallel scripts requires parallel HDF5 as an additional dependency. 

To run the scripts enter : `mpirun -n XX python3 dynamicTomo.py` where XX is the number of cores or GPUs available. \\

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations (available as a thirdparty submodule).  vc
