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
    
    # Source files - find all C++ and CUDA files in gpu/Utils
    sources = []
    gpu_utils_dir = 'tomofusion/gpu/Utils'
    if os.path.exists(gpu_utils_dir):
        sources.extend(glob.glob(os.path.join(gpu_utils_dir, '*.cpp')))
        sources.extend(glob.glob(os.path.join(gpu_utils_dir, '*.cu')))
    
    if not sources:
        print("Warning: No source files found in tomofusion/gpu/Utils/")
        return None
    
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
        'tomofusion.gpu.utils',
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
    
    # Source files - find all C++ and CUDA files in chemistry/Utils
    sources = []
    chem_utils_dir = 'tomofusion/chemistry/Utils'
    if os.path.exists(chem_utils_dir):
        sources.extend(glob.glob(os.path.join(chem_utils_dir, '*.cpp')))
        sources.extend(glob.glob(os.path.join(chem_utils_dir, '*.cu')))
    
    if not sources:
        return None  # No chemistry extension sources found
    
    return Extension(
        'tomofusion.chemistry.utils',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['cudart'],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
        extra_link_args=['-Wl,-rpath,' + os.path.join(CUDA_HOME, 'lib64')],
    )

def get_all_extensions():
    """Dynamically discover and build all submodule extensions"""
    import pybind11  
    extensions = []
    
    # Find all Utils directories with potential extensions
    utils_dirs = []
    for root, dirs, files in os.walk('tomofusion'):
        if 'Utils' in dirs:
            utils_path = os.path.join(root, 'Utils')
            # Check if there are C++ or CUDA source files
            cpp_files = glob.glob(os.path.join(utils_path, '*.cpp'))
            cu_files = glob.glob(os.path.join(utils_path, '*.cu'))
            if cpp_files or cu_files:
                utils_dirs.append((root, utils_path, cpp_files + cu_files))
    
    for module_path, utils_path, sources in utils_dirs:
        # Convert path to module name (e.g., 'tomofusion/gpu' -> 'tomofusion.gpu.utils')
        module_name = module_path.replace('/', '.') + '.utils'
        
        extension = Extension(
            module_name,
            sources=sources,
            include_dirs=[
                np.get_include(),
                pybind11.get_include(), 
                os.path.join(CUDA_HOME, 'include'),
                os.path.join(ASTRA_HOME, 'include'),
                'thirdparty/eigen',
                utils_path,
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, 'lib64'),
                os.path.join(ASTRA_HOME, 'lib'),
                utils_path,
            ],
            libraries=['cudart', 'astra'],
            language='c++',
            extra_compile_args=['-std=c++14', '-O3', '-DWITH_CUDA'],
            extra_link_args=[
                '-Wl,-rpath,' + os.path.join(CUDA_HOME, 'lib64'),
                '-Wl,-rpath,' + os.path.join(ASTRA_HOME, 'lib'),
            ],
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

# Build extensions list
extensions = []

# Option 1: Use dynamic discovery (recommended)
extensions = get_all_extensions()

# Option 2: Manual extension building (if you prefer explicit control)
# gpu_ext = get_cuda_extension()
# if gpu_ext:
#     extensions.append(gpu_ext)
# 
# chem_ext = get_chemistry_extension()
# if chem_ext:
#     extensions.append(chem_ext)

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
    
    # Entry points if you have command-line tools
    # entry_points={
    #     'console_scripts': [
    #         'tomofusion-cli=tomofusion.cli:main',
    #     ],
    # },
)