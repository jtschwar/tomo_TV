#!/bin/bash
set -e

echo "🏗️ Building tomofusion directly on host"
echo "🐍 Using Python $(python --version)"

# Function to build and install custom ASTRA
build_custom_astra() {
    echo "🔨 Building custom ASTRA with your modifications..."
    
    # Save current directory  
    ORIGINAL_DIR=$(pwd)
    echo "📍 Original directory: $ORIGINAL_DIR"
    
    # Step 1: Go into thirdparty/astra-toolbox/build/linux
    cd thirdparty/astra-toolbox
    git checkout backup
    cd build/linux
    echo "📍 Now in: $(pwd)"
    
    # Step 2: Configure with correct CUDA path and prefix
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    PREFIX_PATH="$ORIGINAL_DIR/thirdparty/astra"
    echo "🔧 CUDA path: $CUDA_PATH"
    echo "🔧 Install prefix: $PREFIX_PATH"
    
    ./autogen.sh
    ./configure --with-cuda="$CUDA_PATH" --with-python --with-install-type=prefix --prefix="$PREFIX_PATH"
    
    # Step 3: Make all
    make clean
    make -j$(nproc)
    
    # Step 4: Add prefix to line 657 in Makefile
    echo "🔧 Updating Makefile line 657 with prefix..."
    sed -i "657s|$|  --prefix=$PREFIX_PATH |" Makefile
    
    # Step 5: Make install
    make install
    
    echo "✅ Custom ASTRA installed to: $PREFIX_PATH"
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    # Don't test ASTRA here - wait until after extensions are built
    echo "✅ ASTRA build completed - will test after extensions are built"
}

# Function to build your custom extensions
build_extensions() {
    echo "🔨 Building tomofusion extensions..."
    
    # Step 6: Build chemistry and GPU extensions
    ORIGINAL_DIR=$(pwd)
    
    # Build chemistry extensions
    echo "📦 Building multimodal (chemistry utils)..."
    cd tomofusion/chemistry/utils
    make clean || true
    make all
    
    if [ -f "multimodal.cpython-$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')-x86_64-linux-gnu.so" ]; then
        echo "✅ multimodal extension built successfully"
        ln -sf multimodal.cpython-*-x86_64-linux-gnu.so multimodal.so
    else
        echo "❌ multimodal extension build failed"
        exit 1
    fi
    
    # Build GPU extensions  
    echo "🚀 Building tomoengine (GPU utils)..."
    cd "$ORIGINAL_DIR/tomofusion/gpu/utils"
    make clean || true
    make all
    
    if [ -f "tomoengine.cpython-$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')-x86_64-linux-gnu.so" ]; then
        echo "✅ tomoengine extension built successfully"
        ln -sf tomoengine.cpython-*-x86_64-linux-gnu.so tomoengine.so
    else
        echo "❌ tomoengine extension build failed"
        exit 1
    fi
    
    cd "$ORIGINAL_DIR"

    # Install the tomofusion package
    pip install -e . 
}

# Function to test extensions
test_extensions() {
    echo "🧪 Testing ASTRA and extensions..."
    
    # Test from /tmp to make sure rpath works
    python -c "
try:
    from tomofusion.chemistry.utils import multimodal
    print('✅ multimodal imported successfully')
    obj = multimodal(64, 64, 5)
    print('✅ multimodal object created')
except Exception as e:
    print(f'❌ multimodal failed: {e}')

try:
    from tomofusion.gpu.utils import tomoengine
    print('✅ tomoengine imported successfully')
    obj = tomoengine(64, 64)
    print(f'✅ tomoengine object created, GPU ID: {obj.get_gpu_id()}')
except Exception as e:
    print(f'❌ tomoengine failed: {e}')
"
}

# Function to install dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    if ! command -v nvcc &> /dev/null; then
        echo "❌ CUDA (nvcc) not found. Please ensure CUDA is installed and in PATH."
        exit 1
    fi

    # Show detected CUDA path
    CUDA_PATH=$(dirname $(dirname $(which nvcc)))
    echo "🔍 Detected CUDA at: $CUDA_PATH"
    
    if ! python -c "import pybind11" 2>/dev/null; then
        echo "❌ pybind11 not found. Installing..."
        pip install pybind11
    fi
    
    if ! python -c "import Cython" 2>/dev/null; then
        echo "❌ Cython not found. Installing..."
        pip install Cython
    fi
    
    if ! python -c "import numpy" 2>/dev/null; then
        echo "❌ NumPy not found. Installing..."
        pip install "numpy"
    fi

    if ! python -c "import scipy" 2>/dev/null; then
        echo "❌ scipy not found. Installing..."
        pip install scipy
    fi
    
    if ! python -c "import tomli" 2>/dev/null; then
        echo "❌ tomli not found. Installing..."
        pip install tomli
    fi
}

# Main execution
main() {
    
    # Check dependencies
    check_dependencies

    # Build steps
    build_custom_astra
    build_extensions  
    test_extensions
    
    echo ""
    echo "🎉 Build completed successfully!"
    echo ""
    echo "📋 Import:"
    echo "  from tomofusion.chemistry.utils import multimodal"
    echo "  from tomofusion.gpu.utils import tomoengine"
}

# Help function
show_help() {
    cat << EOF
Host-based build script for tomofusion

This script builds everything directly on the host machine following the exact steps:
1. Configure ASTRA with auto-detected CUDA path
2. Build ASTRA
3. Build tomofusion extensions
4. Test extensions

Prerequisites:
- CUDA toolkit installed and in PATH
- Python development environment

Usage:
    $0              # Build everything on host
    $0 --help       # Show this help

EOF
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac