# Alternative: Single-stage build using CUDA 11.7 manylinux image
# This matches your exact CUDA version from the builder stage

FROM sameli/manylinux2014_x86_64_cuda_11.7

# Install additional build dependencies that might be missing
RUN yum install -y autotools-dev automake libtool boost-devel || \
    yum install -y autoconf automake libtool boost-devel

COPY . /workspace
WORKDIR /workspace

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV ASTRA_HOME=/workspace/thirdparty/astra-toolbox
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ASTRA_HOME}/lib:${LD_LIBRARY_PATH}"

# Install Python dependencies for all Python versions
RUN for PYBIN in /opt/python/cp3{8,9,10,11}-cp3{8,9,10,11}/bin; do \
        "${PYBIN}/pip" install numpy cython six scipy pybind11 matplotlib tqdm h5py scikit-image; \
    done

# Build ASTRA toolbox
RUN cd thirdparty/astra-toolbox/build/linux && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=prefix --prefix=/workspace/thirdparty/astra-toolbox && \
    sed -i "508s/$/  --prefix=\/workspace\/thirdparty\/astra-toolbox /" Makefile && \
    make all && \
    make install

# Build all submodules that have Utils directories with make.inc files
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

# Build wheels for each Python version
RUN for PYBIN in /opt/python/cp3{8,9,10,11}-cp3{8,9,10,11}/bin; do \
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
            --exclude libastra.so || echo "Failed to repair $whl"; \
    done

# Output final wheels
CMD ["cp", "-r", "dist/", "/output/"]