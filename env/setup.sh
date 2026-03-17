#!/usr/bin/env bash
# CS4 Environment Setup Script
# This script creates the conda environment for CS4

echo "Creating CS4 conda environment..."
conda env create -f environment.yaml

echo "Activating CS4 environment..."
conda activate cs4

echo "CS4 environment setup complete!"
echo "To activate in the future, run: conda activate cs4"
