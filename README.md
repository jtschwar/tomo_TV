# tomo_TV

C++ toolbox for Total Varation tomographic reconstruction. Specifically, this repository is designed to study the performance of these algorithms as data is 'dynamically' collected to facilitate real-time analysis of tomograms. 

# Features

2D and 3D reconstruction algorithms implemented purely in C++ wrapped in Python functions.  These scripts can either perform simulations (sim)  or reconstruct experimental (exp) projections . 

# Installation
     
The list of dependencies for python is stored in the requirements text file. They can all be installed with the following code:
   
   `pip install -r requirements.txt`
   
The C++ modules are already precompiled and all the python scripts should currently run. If you would like to make any changes to the C++ modules be sure to recompile. 

tomo_TV uses the Eigen library for C++ linear algebra and sparse computations. Be sure to specify the location of this library whenever compiling the scripts. 
     
# Contribute

Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues

Feel free to open an issue if you have any comments or concerns. 
    
    
