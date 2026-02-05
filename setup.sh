#!/bin/bash
# Quick setup script for the new ONNX converter

echo "🚀 Setting up ONNX Converter with ONNX Runtime GenAI support"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 To start the converter:"
echo "   streamlit run app_new.py"
echo ""
echo "📝 Supported models include:"
echo "   - Qwen3 (NEW!) ✨"
echo "   - Gemma3"
echo "   - Phi4"
echo "   - SmolLM3"
echo "   - Llama, Mistral, and more!"
echo ""
echo "See README_NEW.md for full documentation"
