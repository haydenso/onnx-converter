# ONNX Converter Presets Guide

This guide explains the conversion presets available in the ONNX converter and when to use each one.

## Understanding Output Files

**Important**: The ONNX Runtime GenAI builder typically outputs a single file named `model.onnx` (or `decoder_model.onnx` for decoder-only models). The precision and quantization format is embedded in the ONNX file metadata, not in the filename.

### Common Questions About Output Files

**Q: Why don't I see files like `model_fp16.onnx`, `model_int8.onnx`, or `model_q4.onnx`?**

A: The builder doesn't append suffixes to filenames by default. All presets produce `model.onnx`, but with different internal formats. You can identify the format by:
- The preset you selected
- The file size (quantized models are smaller)
- ONNX metadata inspection tools

**Q: What about the files you mentioned: `model_bnb4.onnx`, `model_fp16.onnx`, `model_int8.onnx`, `model_q4.onnx`, `model_q4f16.onnx`, `model_quantized.onnx`, `model_uint8.onnx`?**

A: These naming conventions are used by some older tools or custom scripts. This converter produces standard ONNX files. Here's how they map to our presets:

| Your filename convention | This converter's preset | Output filename |
|-------------------------|------------------------|-----------------|
| `model.onnx` (FP32) | FP32 - Full Precision | `model.onnx` |
| `model_fp16.onnx` | FP16 - Recommended | `model.onnx` |
| `model_bnb4.onnx` | INT4 - 4-bit Quantized | `model.onnx` |
| `model_int8.onnx` | INT4 + INT8 Activations | `model.onnx` |
| `model_q4.onnx` | INT4 - 4-bit Quantized | `model.onnx` |
| `model_q4f16.onnx` | INT4 + FP16 Activations | `model.onnx` |
| `model_quantized.onnx` | Any INT4 preset | `model.onnx` |
| `model_uint8.onnx` | UINT4 - Asymmetric | `model.onnx` |

**To generate models with different formats**: Run the converter multiple times with different presets and rename the output files yourself if you need to keep multiple versions.

### Example: Creating Multiple Model Variants

If you want to create multiple versions like the naming convention you mentioned:

```bash
# 1. Convert with FP16 preset
streamlit run app.py
# Select "FP16 - Recommended"
# Output: model.onnx
# Rename to: model_fp16.onnx

# 2. Convert with INT4 preset  
streamlit run app.py
# Select "INT4 - 4-bit Quantized"
# Output: model.onnx
# Rename to: model_q4.onnx

# 3. Convert with INT4 + INT8 preset
streamlit run app.py
# Select "INT4 + INT8 Activations"
# Output: model.onnx
# Rename to: model_int8.onnx

# 4. Convert with INT4 + FP16 preset
streamlit run app.py
# Select "INT4 + FP16 Activations"
# Output: model.onnx
# Rename to: model_q4f16.onnx

# etc...
```

Alternatively, you can use the command-line builder directly with different output directories:

```bash
# FP16 version
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.5B-Instruct \
  -o ./output_fp16 \
  -p fp16 \
  -e cuda

# INT4 version
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.5B-Instruct \
  -o ./output_int4 \
  -p int4 \
  -e cuda \
  --extra_options int4_block_size=32 int4_is_symmetric=true int4_accuracy_level=4
```

---

## Available Presets

### 1. FP16 - Recommended (GPU) ⭐
**Default and recommended for most use cases**

- **Precision**: FP16 (half precision)
- **Output**: `model.onnx` (FP16 format)
- **Best for**: 
  - Most models on NVIDIA GPUs
  - General-purpose inference
  - Good balance between size and quality
- **Model size**: ~50% of FP32
- **Quality**: Excellent for most tasks

**Use this when**: Converting any model for GPU inference

---

### 2. FP32 - Full Precision (CPU)
**Highest quality, larger file size**

- **Precision**: FP32 (full precision)
- **Output**: `model.onnx` (FP32 format)
- **Best for**:
  - CPU-only inference
  - Maximum accuracy requirements
  - Older hardware without FP16 support
- **Model size**: 100% (baseline)
- **Quality**: Best possible

**Use this when**: Running on CPU or need maximum precision

---

### 3. BF16 - Brain Float (Gemma/Phi)
**Optimized for specific model families**

- **Precision**: BF16 (brain float 16)
- **Output**: `model.onnx` (BF16 format)
- **Best for**:
  - **Gemma** models (all versions)
  - **Phi** models (Phi-3, Phi-4)
  - Modern GPUs (A100, H100, RTX 30xx+)
- **Model size**: ~50% of FP32
- **Quality**: Better than FP16 for Gemma/Phi models

**Use this when**: Converting Gemma or Phi models, or have modern GPUs

---

### 4. INT4 - 4-bit Quantized
**Smallest size, good quality**

- **Precision**: INT4 (4-bit quantization)
- **Configuration**:
  - Block size: 32
  - Symmetric quantization
  - INT8 activations (accuracy level 4)
- **Output**: `model.onnx` (INT4 quantized)
- **Best for**:
  - Mobile devices
  - Edge deployment
  - Low memory environments
  - Faster inference
- **Model size**: ~25% of FP32
- **Quality**: Good (slight degradation)

**Use this when**: Need smallest possible size or deploying to mobile/edge

---

### 5. INT4 + INT8 Activations
**Balanced quantization**

- **Precision**: INT4 weights + INT8 activations
- **Configuration**:
  - Block size: 32
  - Symmetric quantization
  - INT8 activations (accuracy level 4)
- **Output**: `model.onnx` (INT4/INT8)
- **Best for**:
  - Balanced size/quality tradeoff
  - When INT4 alone loses too much quality
- **Model size**: ~25-30% of FP32
- **Quality**: Better than pure INT4

**Use this when**: INT4 quality isn't good enough but still need small size

---

### 6. INT4 + BF16 Activations
**Quantized Gemma optimization**

- **Precision**: INT4 weights + BF16 activations
- **Configuration**:
  - Block size: 32
  - Symmetric quantization
  - BF16 activations (accuracy level 3)
- **Output**: `model.onnx` (INT4/BF16)
- **Best for**:
  - Gemma models with quantization
  - Phi models with quantization
  - Modern GPUs with limited memory
- **Model size**: ~25-30% of FP32
- **Quality**: Excellent for Gemma/Phi

**Use this when**: Quantizing Gemma or Phi models

---

### 7. INT4 + FP16 Activations
**Standard GPU quantization**

- **Precision**: INT4 weights + FP16 activations
- **Configuration**:
  - Block size: 32
  - Symmetric quantization
  - FP16 activations (accuracy level 2)
- **Output**: `model.onnx` (INT4/FP16)
- **Best for**:
  - Standard GPU inference
  - Most quantized models
- **Model size**: ~25-30% of FP32
- **Quality**: Very good

**Use this when**: Need quantization on GPU with good quality

---

### 8. UINT4 - Asymmetric Quantization
**Alternative quantization method**

- **Precision**: UINT4 (unsigned 4-bit)
- **Configuration**:
  - Block size: 32
  - **Asymmetric** quantization
  - INT8 activations (accuracy level 4)
- **Output**: `model.onnx` (UINT4)
- **Best for**:
  - Alternative to symmetric INT4
  - Some models may perform better
- **Model size**: ~25% of FP32
- **Quality**: Similar to INT4

**Use this when**: Symmetric INT4 doesn't work well for your model

---

### 9. Custom - Manual Configuration
**Full control over all settings**

- **Precision**: Configurable
- **All options**: Fully customizable
- **Best for**:
  - Advanced users
  - Specific requirements
  - Experimentation

**Use this when**: You know exactly what settings you need

---

## Quick Selection Guide

### By Model Type

| Model Family | Recommended Preset | Alternative |
|--------------|-------------------|-------------|
| **Qwen3** | FP16 - Recommended | INT4 + FP16 Activations |
| **Gemma** (all) | BF16 - Brain Float | INT4 + BF16 Activations |
| **Phi** (all) | BF16 - Brain Float | INT4 + BF16 Activations |
| **SmolLM3** | FP16 - Recommended | INT4 - 4-bit Quantized |
| **Llama** | FP16 - Recommended | INT4 + FP16 Activations |
| **Mistral** | FP16 - Recommended | INT4 + FP16 Activations |

### By Hardware

| Hardware | Recommended Preset | Alternative |
|----------|-------------------|-------------|
| **NVIDIA GPU** (RTX 30xx+) | FP16 - Recommended | BF16 - Brain Float |
| **NVIDIA GPU** (older) | FP16 - Recommended | INT4 + FP16 Activations |
| **CPU** | FP32 - Full Precision | INT4 - 4-bit Quantized |
| **Mobile/Edge** | INT4 - 4-bit Quantized | INT4 + INT8 Activations |
| **A100/H100** | BF16 - Brain Float | FP16 - Recommended |

### By Use Case

| Use Case | Recommended Preset |
|----------|-------------------|
| Production inference (GPU) | FP16 - Recommended |
| Production inference (CPU) | FP32 - Full Precision |
| Mobile app | INT4 - 4-bit Quantized |
| Research/development | FP16 - Recommended |
| Minimal storage | INT4 - 4-bit Quantized |
| Maximum accuracy | FP32 - Full Precision |

---

## Understanding the Output Files

All presets generate a primary file named `model.onnx` (or sometimes `decoder_model.onnx` for decoder-only models). The file contains the converted model in the specified precision/quantization format.

The actual format is embedded in the ONNX file metadata and doesn't always reflect in the filename. The builder may also generate additional files:

- `genai_config.json` - ONNX Runtime GenAI configuration
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer
- `special_tokens_map.json` - Special tokens

---

## How Presets Map to Settings

Here's what each preset does internally:

```python
# Example: FP16 - Recommended (GPU)
create_model(
    precision="fp16",
    # No extra quantization options
)

# Example: INT4 + BF16 Activations
create_model(
    precision="int4",
    int4_block_size=32,
    int4_is_symmetric=True,
    int4_accuracy_level=3,  # 3 = BF16 activations
)

# Example: UINT4 - Asymmetric
create_model(
    precision="int4",
    int4_block_size=32,
    int4_is_symmetric=False,  # Asymmetric (UINT4)
    int4_accuracy_level=4,
)
```

### INT4 Accuracy Levels

The `int4_accuracy_level` parameter controls activation precision in quantized models:

| Level | Activation Type | Description |
|-------|----------------|-------------|
| 0 | No constraint | Smallest size, lowest quality |
| 1 | FP32 | Full precision activations |
| 2 | FP16 | Half precision activations |
| 3 | BF16 | Brain float activations |
| 4 | INT8 | 8-bit integer activations (default) |

---

## Tips and Best Practices

1. **Start with FP16**: It's the best balance for most use cases
2. **Use BF16 for Gemma/Phi**: These models are trained with BF16
3. **Try INT4 for mobile**: Significantly smaller with acceptable quality loss
4. **Use FP32 only when needed**: File size is 2x larger than FP16
5. **Test your model**: Different models may perform better with different presets
6. **Check your hardware**: Make sure your target hardware supports the chosen precision

---

## Troubleshooting

### Model is too large
- Try INT4 presets
- Use INT4 + INT8 Activations for balance

### Quality is degraded
- Move from INT4 to FP16
- Try INT4 + FP16/BF16 Activations
- Use FP32 for maximum quality

### Gemma model accuracy issues
- Use BF16 - Brain Float preset
- If quantizing, use INT4 + BF16 Activations

### CUDA out of memory during conversion
- Try INT4 preset
- Use CPU execution provider temporarily
- Close other GPU applications

---

## Advanced: Custom Configurations

When using "Custom - Manual Configuration":

1. Select your base precision (fp16, fp32, bf16, int4)
2. Choose execution provider (cuda, cpu, dml, webgpu)
3. If using INT4:
   - Adjust block size (smaller = more granular, slower)
   - Toggle symmetric/asymmetric
   - Set accuracy level for activations
4. Optional advanced features:
   - Exclude embedding layer
   - Exclude LM head
   - Enable CUDA graph (CUDA only)

---

**Need help?** Check the main README.md or open an issue on GitHub.
