# ONNX Converter - ONNX Runtime GenAI Version

Improved v2 based on https://github.com/huggingface/transformers.js/issues/1361

The original ONNX converter does not support latest Qwen3 and Gemma models. This is a similar streamlit app based on the new ONNX builder script for newer models.

This is an updated version of the ONNX converter that uses the modern **ONNX Runtime GenAI builder** instead of the old Transformers.js approach. This enables support for newer models like Qwen3, Gemma3, Phi4, SmolLM3, and more.

## What Changed?

### Old Version (app.py)
- Used Transformers.js conversion scripts
- Limited to older model architectures (up to Qwen2)
- Required bundled transformers.js repository

### New Version (app.py)
- Uses **ONNX Runtime GenAI builder** from Microsoft
- Supports modern architectures including:
  - **Qwen3** ✨ (your primary need!)
  - Gemma3 (text and multimodal)
  - Phi4
  - SmolLM3
  - And many more (see full list below)
- Direct integration with onnxruntime-genai package
- More advanced quantization options

## Supported Model Architectures

The new converter supports the following architectures:

| Architecture | Model Family |
|-------------|--------------|
| `Qwen3ForCausalLM` | **Qwen3** ✨ |
| `Qwen2ForCausalLM` | Qwen2 |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL |
| `Gemma3ForCausalLM` | Gemma 3 |
| `Gemma2ForCausalLM` | Gemma 2 |
| `GemmaForCausalLM` | Gemma |
| `Phi4MMForCausalLM` | Phi-4 |
| `Phi3ForCausalLM` | Phi-3 |
| `Phi3VForCausalLM` | Phi-3 Vision |
| `PhiMoEForCausalLM` | Phi-3 MoE |
| `SmolLM3ForCausalLM` | SmolLM3 |
| `LlamaForCausalLM` | Llama |
| `MistralForCausalLM` | Mistral |
| `ChatGLMForConditionalGeneration` | ChatGLM |
| `GraniteForCausalLM` | Granite |
| `NemotronForCausalLM` | Nemotron |
| `OlmoForCausalLM` | OLMo |
| `Ernie4_5_ForCausalLM` | Ernie |
| `GptOssForCausalLM` | GPT-OSS |

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Install from Source

If you need the latest builder from the ONNX Runtime GenAI repository:

```bash
# Install base requirements
pip install huggingface_hub streamlit PyYAML torch transformers onnx

# Clone and install onnxruntime-genai
git clone https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai/src/python
pip install -e .
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

Then:
1. Enter a Hugging Face model ID (e.g., `Qwen/Qwen3-0.5B-Instruct`)
2. Select precision (fp16, fp32, bf16, int4)
3. Select execution provider (cuda, cpu, dml, webgpu)
4. Configure advanced options (optional)
5. Click "Start Conversion"

### Command Line Usage (Alternative)

If you have the builder installed, you can also use it directly:

```bash
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.5B-Instruct \
  -o ./output \
  -p fp16 \
  -e cuda \
  -c ./cache_dir
```

## Conversion Options

### Precision
- **fp16**: Half precision (recommended for most GPUs)
- **fp32**: Full precision (CPU or older GPUs)
- **bf16**: BFloat16 (newer GPUs, better for Gemma models)
- **int4**: 4-bit quantization (smallest size, faster inference)

### Execution Providers
- **cuda**: NVIDIA GPUs
- **cpu**: CPU inference
- **dml**: DirectML (Windows GPU)
- **webgpu**: Web browsers with WebGPU support

### Advanced Options (INT4 Quantization)
- **int4_block_size**: Block size for quantization (16, 32, 64, 128, 256)
- **int4_is_symmetric**: Symmetric (int4) vs asymmetric (uint4) quantization
- **int4_accuracy_level**: Accuracy level (0-4, where 4=int8, 3=bf16, 2=fp16, 1=fp32)

### Other Options
- **exclude_embeds**: Remove embedding layer (use inputs_embeds instead)
- **exclude_lm_head**: Remove LM head (output hidden_states instead)
- **enable_cuda_graph**: Enable CUDA graph capture (CUDA only)

## Example: Converting Qwen3

```python
# This is what the Streamlit app does internally:

import onnxruntime_genai as og
from onnxruntime_genai.models.builder import create_model

create_model(
    model_name="Qwen/Qwen3-0.5B-Instruct",
    input_path="",  # Download from HF
    output_dir="./qwen3_onnx",
    precision="fp16",
    execution_provider="cuda",
    cache_dir="./cache",
    hf_token="your_token_here"
)
```

## Using the Converted Model

After conversion, use the model with ONNX Runtime GenAI:

```python
import onnxruntime_genai as og

# Load the model
model = og.Model("./qwen3_onnx")
tokenizer = og.Tokenizer(model)

# Generate text
prompt = "What is the capital of France?"
tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
params.input_ids = tokens

generator = og.Generator(model, params)
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

output_tokens = generator.get_sequence(0)
text = tokenizer.decode(output_tokens)
print(text)
```

## Troubleshooting

### Import Error: Cannot find onnxruntime-genai builder

**Solution**: Install onnxruntime-genai:
```bash
pip install onnxruntime-genai
```

### Model Architecture Not Supported

**Solution**: Check if your model's architecture is in the supported list. You can check the architecture by looking at the model's `config.json` file on Hugging Face.

### CUDA Out of Memory

**Solution**: Try using INT4 quantization or a smaller batch size:
- Use `--precision int4`
- Reduce model size by excluding components with `exclude_embeds` or `exclude_lm_head`

### Gemma Models Lose Accuracy

**Solution**: Gemma models work better with BF16 precision:
```bash
--precision bf16
```

Or for INT4 quantization with BF16 I/O:
```bash
--precision int4 --extra_options use_cuda_bf16=true
```

## Migration Guide

If you're migrating from the old app.py:

| Old (Transformers.js) | New (ONNX Runtime GenAI) |
|----------------------|--------------------------|
| `--quantize` | `--precision int4` |
| `--task <task>` | Auto-detected from model config |
| `--trust_remote_code` | `--extra_options hf_remote=true` (default) |
| `--output_attentions` | Not applicable (different architecture) |
| Output: `models/<model_id>/` | Output: Specified output directory |

## Resources

- [ONNX Runtime GenAI GitHub](https://github.com/microsoft/onnxruntime-genai)
- [ONNX Runtime GenAI Documentation](https://onnxruntime.ai/docs/genai/)
- [Supported Models List](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models)

## License

This converter uses the ONNX Runtime GenAI builder, which is licensed under the MIT License.
