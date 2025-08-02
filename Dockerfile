# Dockerfile for building tomofusion wheels with multiple make.inc files
# Multi-stage build: working environment -> manylinux wheels

# Stage 1: Build environment (your working setup)
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 as builder

RUN apt-get update && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -yq wget git vim autotools-dev automake libtool libboost-all-dev python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python

# Install Python build dependencies
RUN pip install numpy cython six scipy pybind11 matplotlib tqdm h5py scikit-image

COPY . /workspace
WORKDIR /workspace

# Build ASTRA toolbox (your exact process)
RUN cd thirdparty/astra-toolbox/build/linux && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=prefix --prefix=/workspace/thirdparty/astra-toolbox && \
    sed -i "508s/$/  --prefix=\/workspace\/thirdparty\/astra-toolbox /" Makefile && \
    make all && \
    make install

# Build all submodules that have Utils directories with make.inc files
# This approach dynamically finds and builds all submodules
RUN for submodule in $(find tomofusion -name "Utils" -type d); do \
        if [ -f "$submodule/make.inc" ] && [ -f "$submodule/Makefile" ]; then \
            echo "Building $submodule..."; \
            cd /workspace/$submodule && \
            make shared_library 2>/dev/null || echo "shared_library target not found in $submodule"; \
            make all 2>/dev/null || echo "No default target in $submodule"; \
            cd /workspace; \
        else \
            echo "Skipping $submodule (no make.inc or Makefile)"; \
        fi; \
    done

# Alternative approach if you know the specific submodules:
# RUN cd /workspace/tomofusion/gpu/Utils && \
#     make shared_library && \
#     make astra_ctvlib
# 
# RUN cd /workspace/tomofusion/chemistry/Utils && \
#     make shared_library && \
#     make chem_utils

# Stage 2: manylinux wheel builder
FROM quay.io/pypa/manylinux_2_28_x86_64 as wheel-builder

# Install CUDA toolkit
RUN yum install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run && \
    sh cuda_11.7.1_515.65.01_linux.run --silent --toolkit

# Copy built artifacts from stage 1
COPY --from=builder /workspace /workspace
COPY --from=builder /workspace/thirdparty/astra-toolbox /usr/local/astra

WORKDIR /workspace

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV ASTRA_HOME=/usr/local/astra
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ASTRA_HOME}/lib:${LD_LIBRARY_PATH}"

# Build wheels for each Python version
RUN for PYBIN in /opt/python/cp3{8,9,10,11}-cp3{8,9,10,11}/bin; do \
        "${PYBIN}/pip" install numpy cython six scipy pybind11 && \
        "${PYBIN}/pip" wheel . -w wheelhouse/ --no-build-isolation; \
    done

# Repair wheels - exclude CUDA and ASTRA libraries that users should have
RUN for whl in wheelhouse/*.whl; do \
        auditwheel repair "$whl" -w dist/ \
            --exclude libcudart.so \
            --exclude libcufft.so \
            --exclude libcublas.so \
            --exclude libcurand.so \
            --exclude libcusparse.so \
            --exclude libastra.so; \
    done

# Output final wheels
CMD ["cp", "-r", "dist/", "/output/"]
