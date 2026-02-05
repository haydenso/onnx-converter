# Quick Preset Reference Card

## 🎯 Choose Your Preset in 3 Questions

### Question 1: What hardware are you targeting?
- **NVIDIA GPU (RTX 30xx or newer)** → Go to Q2
- **NVIDIA GPU (older)** → **FP16 - Recommended**
- **CPU** → **FP32 - Full Precision**
- **Mobile/Edge device** → **INT4 - 4-bit Quantized**

### Question 2: What model family?
- **Gemma (any version)** → **BF16 - Brain Float**
- **Phi (any version)** → **BF16 - Brain Float**
- **Qwen, Llama, Mistral, other** → **FP16 - Recommended**

### Question 3: Need to save space?
- **Yes, smaller file important** → See Q4
- **No, quality matters most** → Use answer from Q2

### Question 4: Quantization (size reduction)
- **Gemma/Phi + need small size** → **INT4 + BF16 Activations**
- **Other models + need small size** → **INT4 + FP16 Activations**
- **Mobile/Edge + smallest possible** → **INT4 - 4-bit Quantized**

---

## 📊 Preset Comparison Table

| Preset | Size | Quality | Speed | Hardware |
|--------|------|---------|-------|----------|
| FP32 - Full Precision | 100% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | CPU |
| FP16 - Recommended | 50% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GPU ⭐ |
| BF16 - Brain Float | 50% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GPU (Gemma/Phi) |
| INT4 - 4-bit | 25% | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Mobile |
| INT4 + INT8 | 27% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GPU/CPU |
| INT4 + BF16 | 27% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GPU (Gemma/Phi) |
| INT4 + FP16 | 27% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | GPU |
| UINT4 - Asymmetric | 25% | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Alternative |

Size = relative to FP32 baseline

---

## 🔥 Most Common Use Cases

### 1. "I just want to convert a model for my NVIDIA GPU"
→ **FP16 - Recommended (GPU)** ⭐

### 2. "I'm converting a Gemma or Phi model"
→ **BF16 - Brain Float (Gemma/Phi)**

### 3. "I need the smallest possible file for mobile"
→ **INT4 - 4-bit Quantized**

### 4. "I want to run on CPU with best quality"
→ **FP32 - Full Precision (CPU)**

### 5. "I want Gemma but smaller file size"
→ **INT4 + BF16 Activations**

### 6. "I want good quality but smaller than FP16"
→ **INT4 + FP16 Activations**

---

## 🎓 Model-Specific Recommendations

### Qwen3
- **Best**: FP16 - Recommended
- **Small**: INT4 + FP16 Activations
- **CPU**: FP32 - Full Precision

### Gemma (all versions)
- **Best**: BF16 - Brain Float ⭐
- **Small**: INT4 + BF16 Activations
- **CPU**: FP32 - Full Precision

### Phi-3 / Phi-4
- **Best**: BF16 - Brain Float ⭐
- **Small**: INT4 + BF16 Activations
- **CPU**: FP32 - Full Precision

### SmolLM3
- **Best**: FP16 - Recommended
- **Small**: INT4 - 4-bit Quantized ⭐
- **CPU**: FP32 - Full Precision

### Llama / Mistral
- **Best**: FP16 - Recommended
- **Small**: INT4 + FP16 Activations
- **CPU**: FP32 - Full Precision

---

## ⚡ Performance Tips

### Speed Priority
1. INT4 - 4-bit Quantized (fastest)
2. INT4 + FP16 Activations
3. FP16 - Recommended
4. BF16 - Brain Float
5. FP32 - Full Precision (slowest)

### Quality Priority
1. FP32 - Full Precision (best)
2. BF16 - Brain Float (for Gemma/Phi)
3. FP16 - Recommended
4. INT4 + BF16 Activations
5. INT4 + FP16 Activations
6. INT4 + INT8 Activations
7. INT4 - 4-bit Quantized (lowest)

### Size Priority
1. INT4 - 4-bit Quantized (smallest)
2. UINT4 - Asymmetric
3. INT4 + FP16 Activations
4. INT4 + BF16 Activations
5. INT4 + INT8 Activations
6. FP16 - Recommended
7. BF16 - Brain Float
8. FP32 - Full Precision (largest)

---

## 🚨 Common Mistakes to Avoid

❌ **Using FP16 for Gemma/Phi**
→ Use BF16 instead for better accuracy

❌ **Using FP32 for GPU inference**
→ Use FP16 or BF16 instead (2x smaller, same quality)

❌ **Using INT4 without understanding quality loss**
→ Test quality before deploying to production

❌ **Not matching precision to hardware**
→ FP16/BF16 for GPU, FP32 for CPU, INT4 for mobile

❌ **Using asymmetric quantization by default**
→ Stick with symmetric (default) unless you have a reason

---

## 📱 Hardware-Specific Guide

### RTX 3090 / 4090
- **Best**: FP16 - Recommended
- **Gemma/Phi**: BF16 - Brain Float
- **Save VRAM**: INT4 + FP16 Activations

### A100 / H100
- **Best**: BF16 - Brain Float ⭐
- **Alternative**: FP16 - Recommended
- **Save VRAM**: INT4 + BF16 Activations

### CPU (Intel/AMD)
- **Best**: FP32 - Full Precision
- **Faster**: INT4 + INT8 Activations
- **Smallest**: INT4 - 4-bit Quantized

### Mobile (Android/iOS)
- **Best**: INT4 - 4-bit Quantized ⭐
- **Better quality**: INT4 + FP16 Activations
- **Highest quality**: FP16 - Recommended (if VRAM allows)

### Jetson Nano / Edge Devices
- **Best**: INT4 - 4-bit Quantized ⭐
- **Alternative**: INT4 + INT8 Activations

---

## 🔍 Troubleshooting by Symptom

### "Model is too big"
→ Try: INT4 - 4-bit Quantized

### "Quality is bad after conversion"
→ Try: Move up the quality ladder (INT4→FP16→FP32)

### "Gemma model has bad accuracy"
→ Try: BF16 - Brain Float ⭐

### "Out of VRAM during inference"
→ Try: INT4 + FP16 Activations

### "Out of memory during conversion"
→ Try: Use CPU execution provider temporarily

### "Conversion is too slow"
→ This is normal, especially for large models

---

## 📥 Output Files

**All presets produce**: `model.onnx` (or `decoder_model.onnx`)

The precision is embedded in the file, not the filename. 

To create multiple versions:
1. Convert with preset A → rename to `model_a.onnx`
2. Convert with preset B → rename to `model_b.onnx`
3. etc.

---

## 🆘 When to Use "Custom"

Use **Custom - Manual Configuration** when:
- You know exact settings you need
- Experimenting with different configurations
- Following specific model author recommendations
- Debugging conversion issues
- Creating non-standard configurations

For 95% of users, presets are better.

---

**See PRESETS_GUIDE.md for complete documentation**
