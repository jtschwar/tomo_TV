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
ASTRA_HOME = os.environ.get('ASTRA_HOME', '/usr/local/astra')

# Check if paths exist
if not os.path.exists(CUDA_HOME):
    raise RuntimeError(f"CUDA not found at {CUDA_HOME}. Set CUDA_HOME environment variable.")

if not os.path.exists(ASTRA_HOME):
    raise RuntimeError(f"ASTRA not found at {ASTRA_HOME}. Set ASTRA_HOME environment variable.")

def get_cuda_extension():
    """Build the CUDA extension for GPU operations"""
    
    # Include directories
    include_dirs = [
        np.get_include(),
        os.path.join(CUDA_HOME, 'include'),
        os.path.join(ASTRA_HOME, 'include'),
        'thirdparty/eigen',
        'tomofusion/gpu/Utils',
    ]
    
    # Library directories
    library_dirs = [
        os.path.join(CUDA_HOME, 'lib64'),
        os.path.join(ASTRA_HOME, 'lib'),
        'tomofusion/gpu/Utils',
    ]
    
    # Libraries to link against
    libraries = [
        'cudart',
        'cufft', 
        'cublas',
        'curand',
        'cusparse',
        'astra',
    ]
    
    # Source files - update these paths based on your actual structure
    sources = [
        'tomofusion/gpu/Utils/astra_ctvlib.cpp',
        # Add other source files as needed
    ]
    
    # Find all .cu files if any
    cu_sources = glob.glob('tomofusion/gpu/Utils/*.cu')
    sources.extend(cu_sources)
    
    # Compiler flags
    extra_compile_args = [
        '-std=c++14',
        '-O3',
        '-DWITH_CUDA',
    ]
    
    # Linker flags
    extra_link_args = [
        '-Wl,-rpath,' + os.path.join(CUDA_HOME, 'lib64'),
        '-Wl,-rpath,' + os.path.join(ASTRA_HOME, 'lib'),
    ]
    
    return Extension(
        'tomofusion.gpu.astra_ctvlib',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

def get_chemistry_extension():
    """Build chemistry-related extensions if needed"""
    
    # Include directories
    include_dirs = [
        np.get_include(),
        os.path.join(CUDA_HOME, 'include'),
        'thirdparty/eigen',
        'tomofusion/chemistry/Utils',
    ]
    
    # Library directories  
    library_dirs = [
        os.path.join(CUDA_HOME, 'lib64'),
        'tomofusion/chemistry/Utils',
    ]
    
    # Source files - update based on actual structure
    sources = glob.glob('tomofusion/chemistry/Utils/*.cpp')
    cu_sources = glob.glob('tomofusion/chemistry/Utils/*.cu')
    sources.extend(cu_sources)
    
    if not sources:
        return None  # No chemistry extension sources found
    
    return Extension(
        'tomofusion.chemistry.chem_utils',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cudart'],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
        extra_link_args=['-Wl,-rpath,' + os.path.join(CUDA_HOME, 'lib64')],
    )

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

# Build extensions list
extensions = []

# Add GPU extension
gpu_ext = get_cuda_extension()
if gpu_ext:
    extensions.append(gpu_ext)

# Add chemistry extension if it exists
chem_ext = get_chemistry_extension()
if chem_ext:
    extensions.append(chem_ext)

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
    
    # Extensions
    ext_modules=extensions,
    
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
        'License :: OSI Approved :: MIT License',
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
    
    # Entry points if you have command-line tools
    # entry_points={
    #     'console_scripts': [
    #         'tomofusion-cli=tomofusion.cli:main',
    #     ],
    # },
)