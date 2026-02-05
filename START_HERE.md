# 🚀 Start Here - Quick Guide

Welcome to the new ONNX converter with **Qwen3 support**!

## ⚡ 30-Second Quick Start

```bash
# 1. Install (takes ~30 seconds)
./setup.sh

# 2. Run (takes ~5 seconds)
streamlit run app.py

# 3. Convert your Qwen3 model!
# Enter: Qwen/Qwen3-0.5B-Instruct
# Select: fp16, cuda
# Click: Start Conversion
```

## 📚 Documentation

Choose your path:

| Time | File | Description |
|------|------|-------------|
| 2 min | **QUICKSTART.md** | Quick reference with all commands |
| 5 min | **SUMMARY.md** | Overview and what's new |
| 10 min | **README.md** | Complete documentation |
| 15 min | **COMPARISON.md** | Old vs new comparison |
| - | **INDEX.md** | Find anything quickly |

## ✨ What You Get

- ✅ **Qwen3 Support** - Convert Qwen3 models to ONNX
- ✅ **26+ Architectures** - Gemma3, Phi4, SmolLM3, and more
- ✅ **Auto Validation** - Checks model compatibility upfront
- ✅ **Advanced Options** - INT4 quantization, multiple precisions
- ✅ **Better Errors** - Clear messages with suggestions

## 🎯 First Time?

1. Read **QUICKSTART.md** (2 minutes)
2. Run `./setup.sh`
3. Run `streamlit run app.py`
4. Enter a Qwen3 model ID
5. Done! ✅

## 💡 Common Commands

```bash
# Install dependencies
./setup.sh

# Test setup
python test_setup.py

# Run converter (web UI)
streamlit run app.py

# Command line conversion
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.5B-Instruct \
  -o ./output \
  -p fp16 \
  -e cuda
```

## 🆘 Need Help?

- **Installation issues?** → Check README.md "Troubleshooting" section
- **Model not supported?** → Check SUPPORTED_ARCHITECTURES in app.py
- **CUDA errors?** → Try INT4 quantization or CPU execution provider
- **Gemma accuracy?** → Use bf16 precision instead of fp16

## 🎉 Ready?

Run: `streamlit run app.py`

Then enter your Qwen3 model and start converting! 🚀

---

*For complete docs, see README.md*
