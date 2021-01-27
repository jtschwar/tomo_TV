# tomo_TV

C++ toolbox for Total Varation tomographic reconstruction. Specifically, this repository is designed to study the performance of these algorithms as data is 'dynamically' collected to facilitate real-time analysis of tomograms. 

# Features

2D and 3D reconstruction algorithms implemented purely in C++ wrapped in Python functions.  These scripts can either perform simulations (sim)  or reconstruct experimental (exp) projections . 

# Installation

To clone the repositiory run: 

` git clone --recursive https://github.com/jtschwar/tomo_TV.git`
     
The list of python dependencies is stored in the requirements text file. They can all be installed with the following code:
   
   `pip install -r requirements.txt`

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations (available as a thirdparty submodule). 
     
# Contribute

Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues

Feel free to open an issue if you have any comments or concerns. 
    
    
