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
    sed -i "657s/$/  --prefix=$PREFIX_PATH /" Makefile
    
    # Step 5: Make install
    make install
    
    echo "✅ Custom ASTRA installed to: $PREFIX_PATH"
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    # Test ASTRA installation
    python -c "
import astra
print(f'✅ Custom ASTRA v{astra.__version__ if hasattr(astra, \"__version__\") else \"unknown\"} available')
try:
    result = astra.test()
    print('✅ ASTRA tests passed')
except Exception as e:
    print(f'⚠️  ASTRA test issue: {e}')
"
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
}

# Function to test extensions
test_extensions() {
    echo "🧪 Testing extensions from any directory..."
    
    # Test from /tmp to make sure rpath works
    cd /tmp
    python -c "
import sys
import os
# Add the project directory to Python path
project_dir = '/hpc/projects/group.czii/jonathan.schwartz/tmp/pyTomo'
sys.path.insert(0, project_dir)

try:
    from tomofusion.chemistry.utils import multimodal
    print('✅ multimodal imported successfully from /tmp')
    obj = multimodal.multimodal(64, 64, 5)
    print('✅ multimodal object created')
except Exception as e:
    print(f'❌ multimodal failed: {e}')

try:
    from tomofusion.gpu.utils import tomoengine
    print('✅ tomoengine imported successfully from /tmp')
    obj = tomoengine.tomoengine(64, 64)
    print(f'✅ tomoengine object created, GPU ID: {obj.get_gpu_id()}')
except Exception as e:
    print(f'❌ tomoengine failed: {e}')
"
    
    # Return to project directory
    cd /hpc/projects/group.czii/jonathan.schwartz/tmp/pyTomo
}

# Function to create wheel
create_wheel() {
    echo "📦 Creating Python wheel..."
    
    # Step 7: Build and install the wheel
    # Clean any previous builds
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    
    # Build wheel
    python -m build --wheel
    
    echo "📦 Built wheels:"
    ls -la dist/
    
    # Show wheel contents
    if ls dist/*.whl 1> /dev/null 2>&1; then
        WHEEL=$(ls dist/*.whl | head -1)
        echo "🔍 Wheel contents (.so files):"
        unzip -l "$WHEEL" | grep "\.so" || echo "No .so files found in wheel"
    fi
}

# Function to test wheel installation
test_wheel() {
    echo "🧪 Testing wheel installation..."
    
    if [ ! -f "$(ls dist/*.whl | head -1 2>/dev/null)" ]; then
        echo "❌ No wheel found to test"
        return 1
    fi
    
    WHEEL=$(ls dist/*.whl | head -1)
    
    # Install wheel
    pip uninstall tomofusion -y 2>/dev/null || true
    pip install "$WHEEL" --force-reinstall
    
    # Test imports from a different directory
    cd /tmp
    python -c "
import tomofusion
print(f'✅ tomofusion imported from {tomofusion.__file__}')

try:
    from tomofusion.chemistry.utils import multimodal
    print('✅ multimodal imported from wheel')
    obj = multimodal.multimodal(32, 32, 3) 
    print('✅ multimodal object created from wheel')
except Exception as e:
    print(f'❌ multimodal from wheel failed: {e}')

try:
    from tomofusion.gpu.utils import tomoengine
    print('✅ tomoengine imported from wheel')
    obj = tomoengine.tomoengine(32, 32)
    print(f'✅ tomoengine object created from wheel, GPU ID: {obj.get_gpu_id()}')
except Exception as e:
    print(f'❌ tomoengine from wheel failed: {e}')
"
    
    # Return to project directory
    cd /hpc/projects/group.czii/jonathan.schwartz/tmp/pyTomo
}

# Main execution
main() {
    # Check dependencies
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
    
    if ! python -c "import numpy" 2>/dev/null; then
        echo "❌ NumPy not found. Installing..."
        pip install "numpy>=2.0,<3.0"
    fi
    
    # Build steps
    build_custom_astra
    build_extensions  
    test_extensions
    create_wheel
    test_wheel
    
    echo ""
    echo "🎉 Host build completed successfully!"
    echo "📦 Wheel location: dist/"
    echo "🚀 Ready for PyPI upload:"
    echo "  twine upload dist/*.whl"
    echo ""
    echo "📋 Usage:"
    echo "  pip install astra-toolbox  # Install standard ASTRA first"
    echo "  pip install tomofusion     # Then install your package"
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
3. Update Makefile line 657 with prefix
4. Install ASTRA
5. Build tomofusion extensions
6. Create and test wheel

Prerequisites:
- CUDA toolkit installed and in PATH
- Python development environment
- Git repository with your custom ASTRA fork

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