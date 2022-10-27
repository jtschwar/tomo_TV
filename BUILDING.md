To clone the repositiory run: 

` git clone --recursive https://github.com/jtschwar/tomo_TV.git`
     
The list of python dependencies is stored in the requirements text file. They can all be installed with the following code:
   
   `pip install -r requirements.txt`

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations (available as a thirdparty submodule). 

# Compiling GPU Projection Operators and Reconstruction Scripts

For GPU accelerated forward and back-projection operators, we utilize the ASTRA toolbox (https://github.com/astra-toolbox/astra-toolbox). Tomo_tv uses a couple custom written C++ scripts to link the two packages together. Replace the \*.cpp / \*.hpp and follow the compilation directions on the github repository to build the source files. Once all the necessary dependencies are compiled, move to tomo_TV/gpu_3D/Utils to find the Makefile. Open make.inc and specify the paths for the ASTRA, CUDA and HDF5 libraries.

To run the scripts enter : `mpirun -n XX python3 dynamicTomo.py` where XX is the number of cores or GPUs available. \\