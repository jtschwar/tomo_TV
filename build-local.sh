#!/bin/bash
set -e

# Get your local Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_TAG="cp$(echo $PYTHON_VERSION | tr -d '.')"

echo "🐍 Building for Python $PYTHON_VERSION ($PYTHON_TAG)"

# Build the Docker image
docker build -f Dockerfile -t tomofusion-builder .

# Create output directory
mkdir -p dist-local

# Build wheel for your specific Python version
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

# Install in your local Python
if ls dist-local/*.whl 1> /dev/null 2>&1; then
    echo "🔧 Installing in local Python..."
    pip install dist-local/*.whl --force-reinstall
    echo "✅ Installation complete!"
    
    # Test import
    python -c "
import tomofusion
print(f'✅ Successfully imported tomofusion')
try:
    from tomofusion.gpu import reconstructor
    print('✅ GPU module imported')
except Exception as e:
    print(f'⚠️  GPU module: {e}')
"
else
    echo "❌ No wheels were built for Python $PYTHON_VERSION"
    exit 1
fi
