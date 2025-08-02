#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11.setup_helpers import ParallelCompile
import pybind11

# Enable parallel compilation
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

class CustomBuildExt(build_ext):
    def build_extensions(self):
        # Detect CUDA 11 installation (required for your kernels)
        cuda_path = os.environ.get('CUDA_PATH', '/usr/local/cuda-11')
        if not os.path.exists(cuda_path):
            # Try common CUDA 11 locations
            for path in ['/usr/local/cuda-11', '/usr/local/cuda-11.8', '/usr/local/cuda-11.7', '/usr/local/cuda']:
                if os.path.exists(path):
                    cuda_path = path
                    break
        
        if not os.path.exists(cuda_path):
            raise RuntimeError(f"CUDA 11 installation not found. Please set CUDA_PATH environment variable.")
        
        print(f"Using CUDA installation at: {cuda_path}")
        
        # Set CUDA environment variables
        os.environ['CUDA_PATH'] = cuda_path
        os.environ['CUDA_HOME'] = cuda_path
        
        # Verify CUDA version is 11.x
        try:
            result = subprocess.run([f"{cuda_path}/bin/nvcc", "--version"], 
                                  capture_output=True, text=True, check=True)
            if "release 11." not in result.stdout:
                print("WARNING: CUDA version may not be 11.x - your kernels may not work correctly")
                print(f"NVCC output: {result.stdout}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("WARNING: Could not verify CUDA version")
        
        super().build_extensions()

# Define CUDA 11 extensions
def get_cuda_extensions():
    cuda_path = os.environ.get('CUDA_PATH', '/usr/local/cuda-11')
    
    # CUDA 11 specific include and library paths
    cuda_include = [
        f"{cuda_path}/include",
        "/usr/local/include",
        "thirdparty/eigen",  # From your submodules
        pybind11.get_cmake_dir() + "/../../../include"
    ]
    
    cuda_libraries = [
        f"{cuda_path}/lib64",
        f"{cuda_path}/lib",
        "/usr/local/lib"
    ]
    
    # ASTRA paths (built from your submodule)
    astra_path = os.environ.get('ASTRA_PATH', 'thirdparty/astra')
    if os.path.exists(f"{astra_path}/include"):
        astra_include = [f"{astra_path}/include"]
        astra_libraries = [f"{astra_path}/lib"]
    else:
        print("WARNING: ASTRA installation not found - using system paths")
        astra_include = ["/usr/local/include/astra"]
        astra_libraries = ["/usr/local/lib"]
    
    # Compiler flags for CUDA 11
    compile_args = [
        "-O3",
        "-std=c++14",  # CUDA 11 supports C++14
        "-DWITH_CUDA",
        "-DWITH_PYTHON",
        "-fPIC"
    ]
    
    # CUDA 11 specific libraries
    link_args = [
        "-lcudart",
        "-lcublas",
        "-lcufft",
        "-lcusparse",
        "-lastra"
    ]
    
    extensions = []
    
    # Look for your source files
    src_patterns = [
        "src/*.cpp",
        "*.cpp",
        "ctvlib/*.cpp",
        "astra_ctvlib/*.cpp"
    ]
    
    source_files = []
    for pattern in src_patterns:
        source_files.extend(Path(".").glob(pattern))
    
    if not source_files:
        print("WARNING: No C++ source files found. Expected files matching patterns:", src_patterns)
        return []
    
    # Create extensions for each source file or combine them
    for cpp_file in source_files:
        if cpp_file.name.startswith("test_") or cpp_file.name.endswith("_test.cpp"):
            continue  # Skip test files
            
        # Create module name based on file location
        if "astra" in str(cpp_file):
            ext_name = f"tomo_tv.astra_ctvlib"
        else:
            ext_name = f"tomo_tv.{cpp_file.stem}"
        
        print(f"Creating extension: {ext_name} from {cpp_file}")
        
        ext = Pybind11Extension(
            ext_name,
            sources=[str(cpp_file)],
            include_dirs=cuda_include + astra_include,
            library_dirs=cuda_libraries + astra_libraries,
            libraries=["cudart", "cublas", "cufft", "cusparse", "astra"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            cxx_std=14,  # C++14 for CUDA 11 compatibility
        )
        extensions.append(ext)
    
    return extensions

if __name__ == "__main__":
    setup(
        ext_modules=get_cuda_extensions(),
        cmdclass={"build_ext": CustomBuildExt},
    )