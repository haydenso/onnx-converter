#!/usr/bin/env bash
# Quick setup script for the ONNX converter

set -euo pipefail

PYTHON=${PYTHON:-python}
PIP_INSTALL_INDEX=${ORT_INDEX_URL:-"https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/"}

echo "🚀 Setting up ONNX Converter with ONNX Runtime GenAI support"
echo ""

echo "Using python: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "Python version: $($PYTHON --version 2>&1)"
echo ""

echo "📦 Upgrading pip/tools..."
$PYTHON -m pip install --upgrade pip setuptools wheel

echo "📥 Installing requirements from requirements.txt"
$PYTHON -m pip install -r requirements.txt || true

echo "\n🔎 Verifying onnxruntime-genai import..."
if $PYTHON -c "import importlib,sys; importlib.import_module('onnx_ir')" 2>/dev/null; then
    echo "✓ onnx_ir import successful"
else
    echo "⚠ onnx_ir import failed — attempting to install onnxruntime-genai from ORT nightly feed"
    echo "Using index: $PIP_INSTALL_INDEX"
    # Try installing directly from the ORT nightly feed. This may require authentication for private feeds.
    $PYTHON -m pip install --index-url "$PIP_INSTALL_INDEX" onnxruntime-genai || true

    echo "Re-checking import..."
    if $PYTHON -c "import importlib,sys; importlib.import_module('onnx_ir')" 2>/dev/null; then
        echo "✓ onnx_ir import successful after installing from nightly feed"
    else
        echo "✖ Failed to import onnx_ir after trying the nightly feed."
        echo "  - If the feed requires authentication, set ORT_INDEX_URL with credentials or configure pip accordingly."
        echo "  - Example: export ORT_INDEX_URL=\"https://<user>:<PAT>@aiinfra.pkgs.visualstudio.com/.../simple/\""
        echo "  - Alternatively, install from source (see README): git clone https://github.com/microsoft/onnxruntime-genai.git"
    fi
fi

echo "\n✅ Setup finished (may require manual steps if onnxruntime-genai is private)."
echo "To run the app: streamlit run app.py"
