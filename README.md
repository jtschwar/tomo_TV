# tomo_TV

Python and C++ toolbox for tomographic data processing and developing iterative reconstruction algorithms. Specifically, this repository provides a selection of various data models and regularizers for simple python development. Tomo_TV also contains supports experiments where data is 'dynamically' collected to facilitate real-time analysis of tomograms. 

## Features

2D and 3D reconstruction algorithms implemented purely in C++ wrapped in Python functions.  These scripts can either perform simulations (sim) or reconstruct experimental (exp) projections. Available algorithms include:
* Filtered Backprojection (FBP)
* Simultaneous Iterative/Algebraic Reconstruction Technique (SIRT/SART)
* KL-Divergence / Expectation Maximization for Poisson Limited Datasets
* FISTA [doi: 10.1137/080716542](https://epubs.siam.org/doi/10.1137/080716542)
* ASD - POCS [doi: 10.1088/0031-9155/53/17/021](https://iopscience.iop.org/article/10.1088/0031-9155/53/17/021)

We provide a sample jupyter notebook ([demo.ipynb](demo.ipynb)) which outlines the reconstruction process for all these algorithms both with simulated and experimental datasets. 

## Installation

To clone the repositiory and all the core dependencies run the following line in the terminal: 

` git clone --recursive https://github.com/jtschwar/tomo_TV.git`

Instructructions for building can be found in [BUILDING.MD](BUILDING.md).

## Quickstart 

We can either use the traditional non-multimodal reconstruction algorithms:

```python
from tomo_tv.gpu_3D.reconstructor import reconstructor

# Load the Tilt Series and Tilt Angles
# Tilt Series needs to be in (Nx,Ny,Nangles) where Nx is the tilt-axis
# Tilt Angles is a 1D Vector with Nangles elements

# Create Reconstruction object, run reconstruction algorithm and return algorithm
recon = reconstructor(tiltAngles, tiltSeries)
recon.fista()
vol = recon.get_recon()
```

or fused mutli-modal implementation:

```python
from tomo_tv.fused_multi_modal import reconstructor

# Load the Tilt Series and Tilt Angles for ADF and Chemical Signals
# Tilt Series needs to be in (Nx,Ny,Nangles) where Nx is the tilt-axis
# Tilt Angles is a 1D Vector with Nangles elements

chem = {'C': carbon_tilt_series, 'Zn': zn_tilt_series}
recon = reconstructor(adf, adf_tilt_angles, chem, chem_tilt_angles)
recon.data_fusion()
```

## References
If you use tomo_TV for your research, we would appreciate it if you cite to the following papers:

- [Real-time 3D analysis during electron tomography using tomviz](https://www.nature.com/articles/s41467-022-32046-0)
- [Imaging 3D Chemistry at 1 nm resolution with fused multi-modal electron tomography](https://www.nature.com/articles/s41467-024-47558-0)
     
## Contribute

Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues

Feel free to open an issue if you have any comments or concerns. 
    
## Contact

email: [jtschw@umich.edu](jtschw@umich.edu)
website: [https://jtschwar.github.io](https://jtschwar.github.io)
