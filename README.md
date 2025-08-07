# tomo_TV

Python and C++ toolbox for tomographic data processing and developing iterative reconstruction algorithms. Specifically, this repository provides a selection of various data models and regularizers for simple python development. Tomo_TV also contains supports experiments where data is 'dynamically' collected to facilitate real-time analysis of tomograms. 

## Features

2D and 3D reconstruction algorithms implemented purely in C++ wrapped in Python functions.  These scripts can either perform simulations or reconstruct experimental data. Available algorithms include:
* Filtered Backprojection (FBP)
* Simultaneous Iterative/Algebraic Reconstruction Technique (SIRT/SART)
* KL-Divergence / Expectation Maximization for Poisson Limited Datasets
* FISTA [doi: 10.1137/080716542](https://epubs.siam.org/doi/10.1137/080716542)
* ASD - POCS [doi: 10.1088/0031-9155/53/17/021](https://iopscience.iop.org/article/10.1088/0031-9155/53/17/021)

We provide a sample jupyter notebook ([demo.ipynb](demo.ipynb)) which outlines the reconstruction process for all these algorithms both with simulated and experimental datasets. 

## Installation

To build the reconstruction package, clone the repository and run the build script.

```bash
git clone --recursive https://github.com/jtschwar/tomo_TV.git
cd tomo_TV

chmod +x build.sh
./build.sh
```

## Quickstart 

We can either use non-multimodal reconstruction algorithms:

```python
from tomofusion.gpu.reconstructor import TomoGPU

# Load the Tilt Series and Tilt Angles
# Tilt Series needs to be in (Nx,Ny,Nangles) where Nx is the tilt-axis
# Tilt Angles is a 1D Vector with Nangles elements

# Create Reconstruction object, run reconstruction algorithm and return algorithm
tomoengine = TomoGPU(tilt_angles, tilt_series)
tomoengine.fista(Niter=50, lambda_param=1e-1, show_convergence=True)
vol = tomoengine.get_recon() # ( Optional: tomoengine.show_recon() )
```

or fused mutli-modal implementation:

```python
from tomofusion.chemistry.reconstructor import ChemicalTomo
# Add the Chemical Tilt Series to a Dictionary
chem = {'C': carbon_tilt_series, 'Zn': zn_tilt_series}
tomoengine = ChemicalTomo(adf, adf_angles, chem, chem_angles)
tomoengine.data_fusion()
```

## References
If you use tomo_TV for your research, we would appreciate it if you cite to the following papers:

- [Real-time 3D analysis during electron tomography using tomviz](https://www.nature.com/articles/s41467-022-32046-0)
- [Imaging 3D Chemistry at 1 nm resolution with fused multi-modal electron tomography](https://www.nature.com/articles/s41467-024-47558-0)
     
## Contribute

Issue Tracker:  https://github.com/jtschwar/tomo_TV/issues

Feel free to open an issue if you have any comments or concerns. 
    
## Contact

email: [jtschwar@gmail.com](jtschwar@gmail.com)
website: [https://jtschwar.github.io](https://jtschwar.github.io)
