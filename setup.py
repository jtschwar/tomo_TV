#!/usr/bin/env python
"""
Setup script for tomofusion package with CUDA extensions
"""

import os
import sys
import glob
from setuptools import setup, Extension, find_packages
import numpy as np

# Get CUDA and ASTRA paths from environment or use defaults
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
ASTRA_HOME = os.environ.get('ASTRA_HOME', '/workspace/thirdparty/astra-toolbox')

# Check if paths exist
if not os.path.exists(CUDA_HOME):
    raise RuntimeError(f"CUDA not found at {CUDA_HOME}. Set CUDA_HOME environment variable.")

if not os.path.exists(ASTRA_HOME):
    raise RuntimeError(f"ASTRA not found at {ASTRA_HOME}. Set ASTRA_HOME environment variable.")

def get_common_build_config():
    """Get common build configuration shared by all extensions"""
    import pybind11
    
    return {
        'include_dirs': [
            np.get_include(),
            pybind11.get_include(),
            os.path.join(CUDA_HOME, 'include'),
            ASTRA_HOME,                           # Base directory
            os.path.join(ASTRA_HOME, 'include'),  # Include directory
            'thirdparty/eigen',
        ],
        'library_dirs': [
            os.path.join(CUDA_HOME, 'lib64'),
            os.path.join(ASTRA_HOME, 'lib'),
        ],
        'libraries': ['astra', 'cudart', 'cufft', 'cublas', 'curand', 'cusparse', 'z'],  # Add 'z' here, put 'astra' first
        'extra_compile_args': ['-std=c++11', '-O3', '-DWITH_CUDA', '-DASTRA_CUDA', '-fopenmp'],  # Change to c++11, add flags
        'extra_link_args': [
            '-Wl,-rpath,' + os.path.join(CUDA_HOME, 'lib64'),
            '-Wl,-rpath,' + os.path.join(ASTRA_HOME, 'lib'),
            # Remove -lz from here - it goes in 'libraries'
        ],
    }

def get_all_extensions():
    """Dynamically discover and build all submodule extensions"""
    extensions = []
    common_config = get_common_build_config()
    
    # Find all Utils directories with potential extensions
    utils_dirs = []
    for root, dirs, files in os.walk('tomofusion'):
        if 'Utils' in dirs:
            utils_path = os.path.join(root, 'Utils')
            # Check if there are C++ or CUDA source files
            cpp_files = glob.glob(os.path.join(utils_path, '*.cpp'))
            cu_files = glob.glob(os.path.join(utils_path, '*.cu'))

            # Filter out MPI files for now
            cpp_files = [f for f in cpp_files if 'mpi_' not in os.path.basename(f)]

            if cpp_files or cu_files:
                utils_dirs.append((root, utils_path, cpp_files + cu_files))            
    
    for module_path, utils_path, sources in utils_dirs:
        # Convert path to module name (e.g., 'tomofusion/gpu' -> 'tomofusion.gpu.utils')
        module_name = module_path.replace('/', '.') + '.utils'
        
        # Start with common config and add module-specific paths
        include_dirs = common_config['include_dirs'] + [utils_path]
        library_dirs = common_config['library_dirs'] + [utils_path]
        
        extension = Extension(
            module_name,
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=common_config['libraries'],
            language='c++',
            extra_compile_args=common_config['extra_compile_args'],
            extra_link_args=common_config['extra_link_args'],
        )
        extensions.append(extension)
        print(f"Found extension: {module_name} with {len(sources)} source files")
    
    return extensions

# Read version from file
def get_version():
    version_file = os.path.join('tomofusion', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'  # Default version

# Read long description
def get_long_description():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Setup configuration
setup(
    name='tomofusion',
    version=get_version(),
    description='Python and C++ toolbox for tomographic data processing and iterative reconstruction algorithms',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Jonathan Schwartz',
    author_email='jtschw@umich.edu',
    url='https://github.com/jtschwar/tomo_TV',
    
    # Package discovery
    packages=find_packages(),
    
    # Extensions - Use dynamic discovery only
    ext_modules=get_all_extensions(),
    
    # Dependencies
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
        'scikit-image>=0.18.0',
        'h5py>=3.0.0',
        'tqdm>=4.60.0',
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black',
            'flake8',
            'jupyter',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    
    # Keywords
    keywords='tomography reconstruction cuda 3d imaging electron-microscopy',
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)