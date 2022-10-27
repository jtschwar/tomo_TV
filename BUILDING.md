# Building tomo_TV

This project serves as a wrapper around [ASTRA-toolbox](https://github.com/astra-toolbox/astra-toolbox) for GPU acccelerated and custom written parallel C++ reconstraction algorithms. In both cases, we recommend using the GNU GCC compiler, any version above 4.0 will be suitable for compiling these scripts. To install the C++ accelerated scripts:

    cd cpu_3D/Utils/
    make ctvlib
    mkdir paraview-build
    cd paraview-build
    
To compile multi - nodal scripts compile the mpi_ctvlib instead (assuming MPI is available).

# Compiling GPU Accelerated Reconstruction Scripts

For GPU accelerated forward and back-projection operators, we first will need to be in the ASTRA-toolbox folder in `thirdparty/astra-toolbox`. Tomo_tv uses a couple custom written C++ scripts to link the two packages together. Replace the \*.cpp / \*.hpp and follow the compilation directions on the github repository to build the source files. Once all the necessary dependencies are compiled, move to tomo_TV/gpu_3D/Utils to find the Makefile. Open make.inc and specify the paths for the ASTRA, CUDA and HDF5 libraries.

To run the scripts enter : `mpirun -n XX python3 dynamicTomo.py` where XX is the number of cores or GPUs available. \\

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations (available as a thirdparty submodule).  vc