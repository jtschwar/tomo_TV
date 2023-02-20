# tomo_TV

Python and C++ toolbox for tomographic data processing and developing iterative reconstruction algorithms. Specifically, this repository provides a selection of various data models and regularizers for simple python development. Tomo_TV also contains supports experiments where data is 'dynamically' collected to facilitate real-time analysis of tomograms. 

## Features

2D and 3D reconstruction algorithms implemented purely in C++ wrapped in Python functions.  These scripts can either perform simulations (sim) or reconstruct experimental (exp) projections. Available algorithms include:
* Filtered Backprojection (FBP)
* Simultaneous Iterative/Algebraic Reconstruction Technique (SIRT/SART)
* Conjugate Gradient - Least Squares (CGLS)
* KL-Divergence / Expectation Maximization for Poisson Limited Datasets
* FISTA [doi: 10.1137/080716542](https://epubs.siam.org/doi/10.1137/080716542)
* ASD - POCS [doi: 10.1088/0031-9155/53/17/021](https://iopscience.iop.org/article/10.1088/0031-9155/53/17/021)
* (TODO: OGM and FASTA )

We provide a sample jupyter notebook (demo.ipynb) outlining the reconstruction process for all these algorithms.

## Installation

To clone the repositiory and all the core dependencies run the following line in the terminal: 

` git clone --recursive https://github.com/jtschwar/tomo_TV.git`

For GPU accelerated reconstruction algorithms, we recomend using a Linux operating system. C++ accelerated operations is available on all three operating systems (Windows, macOS, and Linux). 

Instructructions for building can be found in [BUILDING.MD](BUILDING.md).

## Multi-GPU Capabilities
tomo_TV can be used by running in parallel across multiple GPU devices on a personal computer or compute nodes in a high-performance computing cluster. In order to initiate a parallel run on multiple GPUs, MPI needs to be available. 

## References
If you use tomo_TV for your research, we would appreciate it if you cite to the following papers:

- [Real-time 3D analysis during electron tomography using tomviz](https://www.nature.com/articles/s41467-022-32046-0)
     
## Contribute

Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues

Feel free to open an issue if you have any comments or concerns. 
    
## Contact

email: [jtschw@umich.edu](jtschw@umich.edu)
website: [https://jtschwar.github.io](https://jtschwar.github.io)
