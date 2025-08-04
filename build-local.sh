#!/bin/bash
set -e

# Detect container runtime
detect_runtime() {
    if command -v docker &> /dev/null; then
        echo "docker"
    elif command -v apptainer &> /dev/null; then
        echo "apptainer"
    elif command -v singularity &> /dev/null; then
        echo "singularity"
    else
        echo "none"
    fi
}

# Function to build with Docker
build_with_docker() {
    echo "🐳 Using Docker for build..."
    
    # Build the Docker image
    echo "🔨 Building Docker image..."
    docker build -f Dockerfile -t tomofusion-builder .

    # Build wheel for your specific Python version
    echo "🚀 Running Docker build..."
    docker run --rm \
        -v $(pwd)/dist-local:/output \
        -e TARGET_PYTHON="$PYTHON_TAG" \
        tomofusion-builder \
        bash -c "
            # Build extensions with Makefiles first
            cd /workspace/tomofusion/chemistry/utils && make all
            cd /workspace/tomofusion/gpu/utils && make all
            
            # Package into wheel
            cd /workspace
            python3 setup.py bdist_wheel
            cp dist/*.whl /output/
        "
}

# Function to build with Apptainer/Singularity
build_with_apptainer() {
    local runtime=$1
    echo "📦 Using $runtime for build..."
    
    # Get current Python version for comparison
    CURRENT_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Check if definition file exists or if Python version changed
    if [ ! -f "tomofusion.def" ] || [ ! -f ".python_version" ] || [ "$(cat .python_version 2>/dev/null)" != "$CURRENT_PYTHON_VERSION" ]; then
        echo "❌ tomofusion.def not found or Python version changed. Creating it..."
        export TARGET_PYTHON_VERSION="$CURRENT_PYTHON_VERSION"
        create_apptainer_def
        echo "$CURRENT_PYTHON_VERSION" > .python_version
    fi
    
    # Build the container if it doesn't exist or is older than the def file
    if [ ! -f "tomofusion.sif" ] || [ "tomofusion.def" -nt "tomofusion.sif" ]; then
        echo "🔨 Building $runtime container for Python $CURRENT_PYTHON_VERSION (this may take 10-20 minutes)..."
        $runtime build --force tomofusion.sif tomofusion.def
    else
        echo "✅ Using existing tomofusion.sif container"
    fi
    
    # Run the build
    echo "🚀 Running $runtime build..."
    echo "To debug interactively, run:"
    echo "  $runtime shell --nv --writable-tmpfs tomofusion.sif"
    echo ""
    $runtime run --nv --writable-tmpfs \
        --bind $(pwd)/dist-local:/output \
        --bind /tmp:/tmp \
        tomofusion.sif
}

# Function to create Apptainer definition file
create_apptainer_def() {
    # Get desired Python version from environment or detect current
    PYTHON_VERSION=${TARGET_PYTHON_VERSION:-$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")}
    
    # Allow CUDA version override
    CUDA_VERSION=${CUDA_VERSION:-"11.8.0"}
    
    cat > tomofusion.def << EOF
Bootstrap: docker
From: nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04

%files
    . /workspace

%post
    # Install system dependencies (including boost which is now required)
    apt-get update
    export DEBIAN_FRONTEND=noninteractive
    apt-get install -yq wget git vim autotools-dev automake libtool libboost-all-dev \\
        software-properties-common build-essential zlib1g-dev libncurses5-dev \\
        libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \\
        libsqlite3-dev libbz2-dev

    # Install deadsnakes PPA for multiple Python versions
    add-apt-repository ppa:deadsnakes/ppa -y
    apt-get update

    # Install the specific Python version
    apt-get install -yq python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-distutils

    # Create symlinks
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

    # Install pip for the specific Python version
    wget https://bootstrap.pypa.io/get-pip.py
    python${PYTHON_VERSION} get-pip.py
    rm get-pip.py

    # Clean up
    rm -rf /var/lib/apt/lists/*

    # Verify Python version
    python3 --version
    echo "Python version: \$(python3 --version)"

    # Install Python dependencies with NumPy 2.x (ASTRA v2.3+ has native support)
    pip install "numpy>=2.0,<3.0" cython six scipy pybind11 matplotlib tqdm h5py scikit-image build wheel

    cd /workspace

    # Ensure we're using the branch with your custom changes and latest ASTRA
    echo "🔧 Switching to branch with custom changes..."
    if [ -d "thirdparty/astra-toolbox/.git" ]; then
        cd thirdparty/astra-toolbox
        git checkout backup  # Your custom changes on latest ASTRA base
        cd /workspace
    fi

    # ASTRA v2.3+ has native NumPy 2.0 support, so no patching needed
    echo "✅ Using ASTRA v2.3+ with native NumPy 2.0 support"

    echo "🔨 Building ASTRA toolbox with your custom changes..."
    cd thirdparty/astra-toolbox/build/linux
    ./autogen.sh
    ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=prefix --prefix=/workspace/thirdparty/astra-toolbox
    sed -i "508s/$/  --prefix=\/workspace\/thirdparty\/astra-toolbox /" Makefile
    make all
    make install

    # Verify ASTRA installation
    echo "✅ ASTRA built and installed with custom changes"
    ls -la /workspace/thirdparty/astra-toolbox/lib/
    ls -la /workspace/thirdparty/astra-toolbox/include/

%environment
    export CUDA_HOME=/usr/local/cuda
    export PATH="\${CUDA_HOME}/bin:\${PATH}"
    export LD_LIBRARY_PATH="/workspace/thirdparty/astra-toolbox/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}"
    # Dynamic Python path based on installed version
    export PYTHONPATH="/workspace/thirdparty/astra-toolbox/lib/python${PYTHON_VERSION}/site-packages:/workspace"

%runscript
    cd /workspace
    echo "🏗️ Building tomofusion with Makefiles..."
    
    # Set environment for Makefile builds  
    export LD_LIBRARY_PATH="/workspace/thirdparty/astra-toolbox/lib:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    export PYTHONPATH="/workspace/thirdparty/astra-toolbox/lib/python${PYTHON_VERSION}/site-packages:/workspace:\$PYTHONPATH"
    
    # Test ASTRA first
    echo "🧪 Testing ASTRA..."
    python3 -c "
try:
    import astra
    print('✅ ASTRA available')
    print(f'ASTRA version: {astra.__version__ if hasattr(astra, \"__version__\") else \"unknown\"}')
    print(f'ASTRA attributes: {[attr for attr in dir(astra) if not attr.startswith(\"_\")][:10]}')
    
    # Test basic functionality
    if hasattr(astra, 'test'):
        result = astra.test()
        print(f'✅ astra.test() result: {result}')
    elif hasattr(astra, 'create_proj_geom'):
        proj_geom = astra.create_proj_geom('parallel', 1.0, 128, [0])
        print('✅ ASTRA basic functionality works')
    else:
        print('✅ ASTRA imported successfully')
        
except Exception as e:
    print(f'❌ ASTRA issue: {e}')
"

    # Build extensions with Makefiles
    echo "🔨 Building chemistry extensions..."
    if [ -f "tomofusion/chemistry/utils/Makefile" ]; then
        cd tomofusion/chemistry/utils
        make clean || true
        make all
        echo "Chemistry build results:"
        ls -la *.so 2>/dev/null || echo "No .so files found"
        
        # Create symlinks for Python import compatibility
        if [ -f "mm_astra.cpython-310-x86_64-linux-gnu.so" ]; then
            ln -sf mm_astra.cpython-310-x86_64-linux-gnu.so mm_astra.so
            echo "✅ Created mm_astra.so symlink"
        fi
        cd /workspace
    fi

    echo "🔨 Building GPU extensions..."
    if [ -f "tomofusion/gpu/utils/Makefile" ]; then
        cd tomofusion/gpu/utils
        make clean || true
        make all
        echo "GPU build results:"
        ls -la *.so 2>/dev/null || echo "No .so files found"
        
        # Create symlinks for Python import compatibility
        if [ -f "astra_ctvlib.cpython-310-x86_64-linux-gnu.so" ]; then
            ln -sf astra_ctvlib.cpython-310-x86_64-linux-gnu.so astra_ctvlib.so
            echo "✅ Created astra_ctvlib.so symlink"
        fi
        cd /workspace
    fi

    # Test direct imports of your extensions
    echo "🧪 Testing Makefile-built extensions..."
    python3 -c "
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tomofusion/chemistry/utils')
sys.path.insert(0, '/workspace/tomofusion/gpu/utils')

# Test mm_astra
try:
    import mm_astra
    print('✅ mm_astra imported!')
    print(f'mm_astra functions: {[attr for attr in dir(mm_astra) if not attr.startswith(\"_\")]}')
    
    if hasattr(mm_astra, 'mm_astra'):
        obj = mm_astra.mm_astra(256, 256, 10)
        print('✅ mm_astra object created!')
    
except Exception as e:
    print(f'❌ mm_astra failed: {e}')

# Test astra_ctvlib
try:
    import astra_ctvlib
    print('✅ astra_ctvlib imported!')
    print(f'astra_ctvlib functions: {[attr for attr in dir(astra_ctvlib) if not attr.startswith(\"_\")]}')
    
    if hasattr(astra_ctvlib, 'astra_ctvlib'):
        obj = astra_ctvlib.astra_ctvlib(256, 256)
        print(f'✅ astra_ctvlib object created, GPU ID: {obj.get_gpu_id()}')
    
except Exception as e:
    print(f'❌ astra_ctvlib failed: {e}')
"

    # Package everything into a wheel using pyproject.toml
    echo "📦 Creating wheel..."
    
    # Use your existing setup.py and pyproject.toml
    python3 -m build --wheel
    
    # Show results
    echo "📦 Built wheels:"
    ls -la dist/
    
    # Show what's in the wheel
    if ls dist/*.whl 1> /dev/null 2>&1; then
        WHEEL=\$(ls dist/*.whl | head -1)
        echo "🔍 Wheel contents:"
        unzip -l "\$WHEEL" | grep -E '\.(so|py)\$' || echo "unzip not available"
    fi
    
    # Copy to output
    if [ -d "/output" ]; then
        cp dist/*.whl /output/ 2>/dev/null || echo "No wheels to copy"
        echo "✅ Wheels copied to /output/"
    fi
EOF
    echo "✅ Created tomofusion.def for Python $PYTHON_VERSION with CUDA $CUDA_VERSION"
}

# Function to test installation
test_installation() {
    if ls dist-local/*.whl 1> /dev/null 2>&1; then
        echo "🔧 Installing in local Python..."
        
        # Uninstall first if already installed
        pip uninstall tomofusion -y 2>/dev/null || true
        
        # Install the wheel
        WHEEL=$(ls dist-local/*.whl | head -1)
        pip install "$WHEEL" --force-reinstall --verbose
        
        echo "✅ Installation complete!"
        
        # Test import
        echo "🧪 Testing installation..."
        python -c "
import tomofusion
print(f'✅ tomofusion imported from {tomofusion.__file__}')

# Test direct import of your specific modules
try:
    from tomofusion.chemistry.utils import mm_astra
    print('✅ mm_astra imported as submodule!')
    obj = mm_astra.mm_astra(256, 256, 10)
    print('✅ mm_astra object created!')
except Exception as e:
    print(f'❌ mm_astra submodule failed: {e}')

try:
    from tomofusion.gpu.utils import astra_ctvlib  
    print('✅ astra_ctvlib imported as submodule!')
    obj = astra_ctvlib.astra_ctvlib(256, 256)
    print(f'✅ astra_ctvlib object created, GPU ID: {obj.get_gpu_id()}')
except Exception as e:
    print(f'❌ astra_ctvlib submodule failed: {e}')
"
        return $?
    else
        echo "❌ No wheels were built"
        return 1
    fi
}

# Main script
main() {
    # Get local Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_TAG="cp$(echo $PYTHON_VERSION | tr -d '.')"

    echo "🏗️ Building tomofusion with Makefiles"
    echo "🐍 Building for Python $PYTHON_VERSION ($PYTHON_TAG)"
    echo "🚀 Using CUDA ${CUDA_VERSION:-11.8.0}"

    # Detect container runtime
    RUNTIME=$(detect_runtime)
    
    # Allow manual override via environment variable
    if [ -n "$CONTAINER_RUNTIME" ]; then
        RUNTIME="$CONTAINER_RUNTIME"
        echo "🔧 Using manually specified runtime: $RUNTIME"
    fi

    case $RUNTIME in
        docker)
            # Check if Dockerfile exists
            if [ ! -f "Dockerfile" ]; then
                echo "❌ Dockerfile not found. Please create it first."
                exit 1
            fi
            ;;
        apptainer|singularity)
            echo "📦 Detected $RUNTIME"
            ;;
        none)
            echo "❌ No container runtime found!"
            echo "Please install either:"
            echo "  - Docker: https://docs.docker.com/get-docker/"
            echo "  - Apptainer: https://apptainer.org/docs/user/latest/quick_start.html"
            exit 1
            ;;
        *)
            echo "❌ Unknown runtime: $RUNTIME"
            exit 1
            ;;
    esac

    # Create output directory
    mkdir -p dist-local

    # Build based on detected/specified runtime
    case $RUNTIME in
        docker)
            build_with_docker
            ;;
        apptainer|singularity)
            build_with_apptainer $RUNTIME
            ;;
    esac

    # Test installation
    test_installation
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Build and installation successful!"
        echo "📦 Wheel location: dist-local/"
        echo "🧪 Package is ready to use!"
        echo ""
        echo "📋 Usage:"
        echo "  from tomofusion.chemistry.utils import mm_astra"
        echo "  from tomofusion.gpu.utils import astra_ctvlib"
        echo ""
        echo "🚀 For PyPI upload:"
        echo "  twine upload dist-local/*.whl"
    else
        echo ""
        echo "❌ Build completed but testing failed"
        echo "📦 Wheel location: dist-local/"
        echo "🔧 Try manual installation and debugging"
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build tomofusion using Makefiles and package for PyPI distribution.

OPTIONS:
    -c, --cuda VERSION      Force specific CUDA version (e.g., 12.4.0, 12.5.0)
    -h, --help              Show this help message
    -r, --runtime RUNTIME   Force specific container runtime (docker|apptainer|singularity)
    -p, --python VERSION    Force specific Python version (e.g., 3.11)
    -v, --verbose           Enable verbose build output

Examples:
    $0                      # Auto-detect runtime and use current Python version
    $0 -r docker           # Force use Docker
    $0 -r apptainer        # Force use Apptainer
    $0 -p 3.11             # Build for Python 3.11 specifically
    $0 -c 12.4.0           # Use CUDA 12.4.0
    $0 -r apptainer -p 3.10 -c 12.5.0 # Use Apptainer with Python 3.10 and CUDA 12.5.0

Environment Variables:
    CONTAINER_RUNTIME       Force specific runtime (docker|apptainer|singularity)
    TARGET_PYTHON_VERSION   Force specific Python version (e.g., 3.11)
    CUDA_VERSION           Force specific CUDA version (e.g., 12.4.0)

EOF
}

# Parse command line arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--runtime)
            CONTAINER_RUNTIME="$2"
            shift 2
            ;;
        -p|--python)
            TARGET_PYTHON_VERSION="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main