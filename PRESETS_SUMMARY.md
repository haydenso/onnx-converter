# Presets Integration Summary

## What Was Added

### 1. **Preset Selector** in Streamlit UI
- Added a dropdown with 9 preset options
- Presets automatically configure precision and quantization settings
- Shows description and "best for" information for each preset
- Locks manual settings when using a preset (unlock with "Custom" option)

### 2. **Available Presets**

| # | Preset Name | Precision | Config | Use Case |
|---|-------------|-----------|--------|----------|
| 1 | **FP16 - Recommended (GPU)** | fp16 | Default | Most models, NVIDIA GPUs ⭐ |
| 2 | **FP32 - Full Precision (CPU)** | fp32 | Default | CPU inference, max accuracy |
| 3 | **BF16 - Brain Float (Gemma/Phi)** | bf16 | Default | Gemma, Phi, A100/H100 GPUs |
| 4 | **INT4 - 4-bit Quantized** | int4 | bs=32, sym, acc=4 | Mobile, edge, low memory |
| 5 | **INT4 + INT8 Activations** | int4 | bs=32, sym, acc=4 | Balanced quantization |
| 6 | **INT4 + BF16 Activations** | int4 | bs=32, sym, acc=3 | Gemma/Phi quantized |
| 7 | **INT4 + FP16 Activations** | int4 | bs=32, sym, acc=2 | Standard GPU quantization |
| 8 | **UINT4 - Asymmetric** | int4 | bs=32, asym, acc=4 | Alternative quantization |
| 9 | **Custom - Manual Configuration** | fp16 | Customizable | Advanced users |

**Legend**: bs=block_size, sym=symmetric, asym=asymmetric, acc=accuracy_level

### 3. **UI Enhancements**
- Info box showing preset description
- "Best for" caption showing use cases
- Expected output filename display
- Locked/unlocked controls based on preset selection
- Visual indication in Advanced Options section

### 4. **Documentation**
- **PRESETS_GUIDE.md**: Complete guide to all presets (6000+ words)
  - Detailed explanation of each preset
  - When to use each one
  - Quick selection guides by model, hardware, and use case
  - Troubleshooting tips
  - Advanced configuration examples
  
- **README.md**: Updated with preset information
  - Table of presets
  - Link to detailed guide
  
- **PRESETS_SUMMARY.md**: This file - quick reference

## How Users Interact With Presets

### Simple Flow (Recommended)
1. Enter model ID
2. Select a preset from dropdown (default: FP16 - Recommended)
3. Choose execution provider
4. Click "Start Conversion"

### Advanced Flow
1. Enter model ID
2. Select "Custom - Manual Configuration" preset
3. Manually set precision, INT4 options, etc.
4. Choose execution provider
5. Configure advanced options
6. Click "Start Conversion"

## Mapping Your Original Question

You asked about these model files:
```
model.onnx
model_bnb4.onnx
model_fp16.onnx
model_int8.onnx
model_q4.onnx
model_q4f16.onnx
model_quantized.onnx
model_uint8.onnx
```

### How They Map to Presets

| Your Filename | Our Preset | Settings |
|---------------|------------|----------|
| `model.onnx` (FP32) | FP32 - Full Precision | precision="fp32" |
| `model_fp16.onnx` | FP16 - Recommended | precision="fp16" |
| `model_bnb4.onnx` | INT4 - 4-bit Quantized | precision="int4", bs=32, sym |
| `model_int8.onnx` | INT4 + INT8 Activations | precision="int4", acc_level=4 |
| `model_q4.onnx` | INT4 - 4-bit Quantized | precision="int4" |
| `model_q4f16.onnx` | INT4 + FP16 Activations | precision="int4", acc_level=2 |
| `model_quantized.onnx` | Any INT4 preset | precision="int4" |
| `model_uint8.onnx` | UINT4 - Asymmetric | precision="int4", asym |

**Important**: All our presets output to `model.onnx`. The format is embedded in the file, not the filename. To create multiple variants, run the converter multiple times and rename the outputs yourself.

## Code Changes

### Modified File
- **app.py** (lines 526-702)
  - Added `PRESETS` dictionary with all configurations
  - Added preset selector dropdown
  - Added info/caption displays
  - Modified precision selectbox to use preset defaults
  - Modified INT4 options to use preset defaults
  - Added disabled state for locked controls

### New Files
- **PRESETS_GUIDE.md** - Complete preset documentation
- **PRESETS_SUMMARY.md** - This file

### Updated Files
- **README.md** - Added preset table and reference

## Technical Details

### Preset Configuration Structure
```python
PRESETS = {
    "Preset Name": {
        "precision": "fp16" | "fp32" | "bf16" | "int4",
        "description": "Human-readable description",
        "int4_block_size": 16 | 32 | 64 | 128 | 256,
        "int4_is_symmetric": True | False,
        "int4_accuracy_level": 0 | 1 | 2 | 3 | 4,
        "best_for": "Use case description"
    }
}
```

### How Presets Work
1. User selects preset from dropdown
2. App reads preset config from `PRESETS` dict
3. UI controls are set to preset values
4. If not "Custom", controls are disabled (locked)
5. On conversion, preset values are passed to builder

### INT4 Accuracy Levels Explained
```python
0 = No constraint (smallest, lowest quality)
1 = FP32 activations (full precision)
2 = FP16 activations (half precision)
3 = BF16 activations (brain float)
4 = INT8 activations (8-bit integer, default)
```

## Testing Recommendations

1. **Test UI Flow**
   ```bash
   streamlit run app.py
   ```
   - Verify preset dropdown appears
   - Check that descriptions show correctly
   - Confirm controls lock/unlock based on preset

2. **Test Each Preset** (optional, time-consuming)
   - Pick a small model (e.g., `Qwen/Qwen2-0.5B`)
   - Try each preset
   - Verify output files are created
   - Compare file sizes (INT4 should be ~25% of FP32)

3. **Test Custom Mode**
   - Select "Custom - Manual Configuration"
   - Verify all controls are enabled
   - Try different manual settings

## Quick Start Examples

### Example 1: Convert Qwen3 with FP16 (recommended)
1. Run: `streamlit run app.py`
2. Enter: `Qwen/Qwen3-0.5B-Instruct`
3. Preset: `FP16 - Recommended (GPU)` (default)
4. Provider: `cuda`
5. Click: "Start Conversion"

### Example 2: Convert Gemma3 with BF16
1. Run: `streamlit run app.py`
2. Enter: `google/gemma-2-2b-it`
3. Preset: `BF16 - Brain Float (Gemma/Phi)`
4. Provider: `cuda`
5. Click: "Start Conversion"

### Example 3: Convert for Mobile (INT4)
1. Run: `streamlit run app.py`
2. Enter: `HuggingFaceTB/SmolLM2-135M-Instruct`
3. Preset: `INT4 - 4-bit Quantized`
4. Provider: `cpu`
5. Click: "Start Conversion"

### Example 4: Custom Configuration
1. Run: `streamlit run app.py`
2. Enter: Your model ID
3. Preset: `Custom - Manual Configuration`
4. Precision: Choose manually
5. Provider: Choose manually
6. Advanced Options: Configure as needed
7. Click: "Start Conversion"

## Benefits of This Implementation

✅ **User-Friendly**: One-click preset selection
✅ **Educational**: Shows what each preset does
✅ **Flexible**: Can still manually configure everything
✅ **Safe**: Locks controls to prevent mistakes with presets
✅ **Well-Documented**: Complete guide for all presets
✅ **Best Practices**: Recommends optimal settings per use case

## Next Steps

1. **Test the UI**: Run `streamlit run app.py` and try the presets
2. **Read the guides**: Check `PRESETS_GUIDE.md` for detailed info
3. **Try converting a model**: Test with a small model first
4. **Share feedback**: Let me know if any adjustments are needed

---

**Need help?** Check:
- **PRESETS_GUIDE.md** for detailed preset information
- **README.md** for general usage
- **app.py** lines 526-702 for implementation details
