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
            # Find the wheel that matches your Python version
            WHEEL=\$(ls dist/*${TARGET_PYTHON}*.whl 2>/dev/null | head -1)
            if [ -n \"\$WHEEL\" ]; then
                echo \"📦 Found wheel: \$WHEEL\"
                cp \"\$WHEEL\" /output/
            else
                echo \"❌ No wheel found for $PYTHON_TAG\"
                echo \"Available wheels:\"
                ls dist/*.whl || echo \"No wheels found\"
                exit 1
            fi
        "
}

# Function to build with Apptainer/Singularity
build_with_apptainer() {
    local runtime=$1
    echo "📦 Using $runtime for build..."
    
    # Check if definition file exists
    if [ ! -f "tomofusion.def" ]; then
        echo "❌ tomofusion.def not found. Creating it..."
        create_apptainer_def
    fi
    
    # Build the container if it doesn't exist or is older than the def file
    if [ ! -f "tomofusion.sif" ] || [ "tomofusion.def" -nt "tomofusion.sif" ]; then
        echo "🔨 Building $runtime container (this may take 10-20 minutes)..."
        $runtime build --force tomofusion.sif tomofusion.def
    else
        echo "✅ Using existing tomofusion.sif container"
    fi
    
    # Run the build
    echo "🚀 Running $runtime build..."
    $runtime run --nv --writable-tmpfs \
        --bind $(pwd)/dist-local:/output \
        tomofusion.sif
}

# Function to create Apptainer definition file
create_apptainer_def() {
    cat > tomofusion.def << 'EOF'
Bootstrap: docker
From: nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

%files
    . /workspace

%post
    # Install system dependencies
    apt-get update
    export DEBIAN_FRONTEND=noninteractive
    apt-get install -yq wget git vim autotools-dev automake libtool libboost-all-dev python3-pip
    rm -rf /var/lib/apt/lists/*
    ln -s /usr/bin/python3 /usr/bin/python

    # Install Python dependencies
    pip install numpy cython six scipy pybind11 matplotlib tqdm h5py scikit-image build

    # Set working directory
    cd /workspace

    # Build ASTRA toolbox
    cd thirdparty/astra-toolbox/build/linux
    ./autogen.sh
    ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=prefix --prefix=/workspace/thirdparty/astra-toolbox
    sed -i "508s/$/  --prefix=\/workspace\/thirdparty\/astra-toolbox /" Makefile
    make all
    make install

    # Build all submodules dynamically
    cd /workspace
    for submodule in $(find tomofusion -name "Utils" -type d 2>/dev/null); do
        if [ -f "$submodule/make.inc" ] && [ -f "$submodule/Makefile" ]; then
            echo "Building $submodule..."
            cd "/workspace/$submodule"
            make shared_library 2>/dev/null || echo "shared_library target not found in $submodule"
            make all 2>/dev/null || echo "No default target in $submodule"
            cd /workspace
        else
            echo "Skipping $submodule (no make.inc or Makefile)"
        fi
    done

%environment
    export CUDA_HOME=/usr/local/cuda
    export ASTRA_HOME=/workspace/thirdparty/astra-toolbox
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${ASTRA_HOME}/lib:${LD_LIBRARY_PATH}"

%runscript
    cd /workspace
    echo "🏗️  Building Python wheel..."
    
    # Build wheel using setup.py directly
    python setup.py bdist_wheel
    
    # Show results
    echo "📦 Built wheels:"
    ls -la dist/
    
    # Copy to output if mounted
    if [ -d "/output" ]; then
        cp dist/*.whl /output/ 2>/dev/null || echo "No wheels to copy"
        echo "✅ Wheels copied to /output/"
    fi
EOF
    echo "✅ Created tomofusion.def"
}

# Function to test installation
test_installation() {
    if ls dist-local/*.whl 1> /dev/null 2>&1; then
        echo "🔧 Installing in local Python..."
        pip install dist-local/*.whl --force-reinstall
        echo "✅ Installation complete!"
        
        # Test import
        echo "🧪 Testing installation..."
        python -c "
import tomofusion
print(f'✅ Successfully imported tomofusion')

# Test GPU module
try:
    from tomofusion.gpu.reconstructor import reconstructor
    print('✅ GPU module imported')
except Exception as e:
    print(f'⚠️  GPU module: {e}')

# Test fused multi-modal
try:
    from tomofusion.fused_multi_modal.reconstructor import reconstructor as FusedReconstructor
    print('✅ Fused multi-modal module imported')
except Exception as e:
    print(f'⚠️  Fused multi-modal: {e}')
"
        return 0
    else
        echo "❌ No wheels were built for Python $PYTHON_VERSION"
        return 1
    fi
}

# Main script
main() {
    # Get local Python version
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_TAG="cp$(echo $PYTHON_VERSION | tr -d '.')"

    echo "🏗️  Building tomofusion locally"
    echo "🐍 Building for Python $PYTHON_VERSION ($PYTHON_TAG)"

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
    else
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build tomofusion locally using Docker or Apptainer/Singularity.

OPTIONS:
    -h, --help              Show this help message
    -r, --runtime RUNTIME   Force specific container runtime (docker|apptainer|singularity)

Examples:
    $0                      # Auto-detect runtime and build
    $0 -r docker           # Force use Docker
    $0 -r apptainer        # Force use Apptainer

Environment Variables:
    CONTAINER_RUNTIME       Force specific runtime (docker|apptainer|singularity)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -r|--runtime)
            CONTAINER_RUNTIME="$2"
            shift 2
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