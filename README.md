# tomo_TV

C++ toolbox for Total Varation tomographic reconstruction. Specifically, this repository is designed to study the performance of these algorithms as data is 'dynamically' collected. 

# Features

2D and 3D reconstruction algorithms implemented purely in C++ (cTV) and C++ wrapped in Python (pycTV).  As of right now, only pycTV supports both 2D and 3D datasets. 

# Installation

   cTV uses OpenCV to import and save images and the Eigen library for linear algebra operations. Eigen is the only library included in the Dependencies folder so OpenCV needs to be compiled first in order to build cTV. Cmake creates the make file. if you are in the cTV directory run these commands to build cTV.

    `cmake .
     make`
     
   The list of dependencies for pycTV is stored in the requirements text file. They can all be installed with the following code:
   
   `pip install -r requirements.txt`
   
   The C++ modules are already precompiled and all the python scripts should currently run. If you would like to make any changes to the C++ modules be sure to recompile. 
     
# Contribute

•   Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues
    Feel free to open an issue if you have any comments or concerns. 
    
    
