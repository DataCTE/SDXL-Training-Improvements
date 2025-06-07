#!/bin/bash
# SDXL Training Setup Script

echo "==================================="
echo "SDXL Training Framework Setup"
echo "==================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install xformers for memory efficiency
echo "Installing xformers..."
pip install xformers

# Create necessary directories
echo "Creating directories..."
mkdir -p data/train
mkdir -p checkpoints
mkdir -p output

# Download example configuration
echo "Setting up configuration..."
if [ ! -f config.yaml ]; then
    echo "✓ config.yaml already exists"
else
    echo "✓ Created config.yaml"
fi

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Place your training images in: data/train/"
echo "2. Add corresponding .txt caption files"
echo "3. Edit config.yaml to customize settings"
echo "4. Run training: python train.py"
echo ""
echo "To activate the environment in the future:"
echo "source venv/bin/activate"