# Quick Reference Card 🚀

## Installation (30 seconds)
```bash
pip install -r requirements.txt
```

## Test (10 seconds)
```bash
python test_setup.py
```

## Run (5 seconds)
```bash
streamlit run app.py
```

---

## Convert Qwen3 Models

### Via Web UI (Recommended)
1. Start app: `streamlit run app.py`
2. Enter: `Qwen/Qwen3-0.5B-Instruct`
3. Select: `fp16` + `cuda`
4. Click: **Start Conversion**

### Via Command Line
```bash
python -m onnxruntime_genai.models.builder \
  -m Qwen/Qwen3-0.5B-Instruct \
  -o ./qwen3_onnx \
  -p fp16 \
  -e cuda
```

---

## Precision Options

| Option | Size | Speed | Quality | Use Case |
|--------|------|-------|---------|----------|
| `fp32` | Large | Slow | Best | CPU inference |
| `fp16` | Medium | Fast | Good | GPU (most common) |
| `bf16` | Medium | Fast | Good | Gemma models |
| `int4` | Small | Fastest | OK | Mobile/edge |

**Recommendation**: Use `fp16` for CUDA, `int4` for CPU/mobile

---

## Execution Providers

| Provider | Hardware | Notes |
|----------|----------|-------|
| `cuda` | NVIDIA GPU | Most common for GPU |
| `cpu` | CPU | Use fp32 or int4 |
| `dml` | Windows GPU | DirectML (AMD/Intel) |
| `webgpu` | Browser | Web deployment |

**Recommendation**: Use `cuda` if you have NVIDIA GPU

---

## Quick Examples

### Qwen3 FP16 (Default)
```bash
streamlit run app.py
# Enter: Qwen/Qwen3-0.5B-Instruct
# Select: fp16, cuda
```

### Qwen3 INT4 Quantized (Smaller)
```bash
streamlit run app.py
# Enter: Qwen/Qwen3-0.5B-Instruct
# Select: int4, cuda
# Advanced: block_size=32, symmetric=true
```

### Qwen3 CPU (No GPU)
```bash
streamlit run app.py
# Enter: Qwen/Qwen3-0.5B-Instruct
# Select: fp32, cpu
```

### Gemma3 BF16 (Better Quality)
```bash
streamlit run app.py
# Enter: google/gemma-3-7b
# Select: bf16, cuda
```

---

## Common Issues

### "Cannot find builder"
```bash
pip install onnxruntime-genai
```

### "Model not supported"
Check architecture in model's `config.json`. Must be one of:
- Qwen3ForCausalLM ✓
- Gemma3ForCausalLM ✓
- Phi4MMForCausalLM ✓
- etc. (see SUPPORTED_ARCHITECTURES in app.py)

### "CUDA out of memory"
Use INT4 quantization:
```bash
# Select: int4 instead of fp16
```

### "Gemma model loses accuracy"
Use BF16:
```bash
# Select: bf16 instead of fp16
```

---

## File Locations

### Input
```
Your Qwen3 model on Hugging Face
↓
https://huggingface.co/Qwen/Qwen3-0.5B-Instruct
```

### Output (Local)
```
./onnx_models/Qwen_Qwen3-0.5B-Instruct/
├── model.onnx
├── genai_config.json
├── config.json
├── tokenizer.json
└── README.md
```

### Output (Hugging Face)
```
https://huggingface.co/YOUR_USERNAME/Qwen3-0.5B-Instruct-ONNX
```

---

## Using Converted Model

### Python
```python
import onnxruntime_genai as og

model = og.Model("./qwen3_onnx")
tokenizer = og.Tokenizer(model)

prompt = "Hello, how are you?"
tokens = tokenizer.encode(prompt)

params = og.GeneratorParams(model)
params.set_search_options(max_length=200)
params.input_ids = tokens

generator = og.Generator(model, params)
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()

output = tokenizer.decode(generator.get_sequence(0))
print(output)
```

### C++
```cpp
#include <onnxruntime_genai.h>

auto model = OgaModel::Create("./qwen3_onnx");
auto tokenizer = OgaTokenizer::Create(*model);

auto input = tokenizer->Encode("Hello, how are you?");
auto params = OgaGeneratorParams::Create(*model);
params->SetSearchOption("max_length", 200);
params->SetInputSequences(input);

auto generator = OgaGenerator::Create(*model, *params);
while (!generator->IsDone()) {
  generator->ComputeLogits();
  generator->GenerateNextToken();
}

auto output = tokenizer->Decode(generator->GetSequence(0));
```

---

## Supported Models (Top 10)

1. **Qwen3** - `Qwen/Qwen3-*` ⭐
2. **Qwen2** - `Qwen/Qwen2-*`
3. **Gemma3** - `google/gemma-3-*`
4. **Phi4** - `microsoft/phi-4-*`
5. **SmolLM3** - `HuggingFaceTB/SmolLM3-*`
6. **Llama** - `meta-llama/Llama-*`
7. **Mistral** - `mistralai/Mistral-*`
8. **Phi-3** - `microsoft/Phi-3-*`
9. **Granite** - `ibm-granite/granite-*`
10. **Nemotron** - `nvidia/Nemotron-*`

---

## Cheat Sheet

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Test | `python test_setup.py` |
| Run UI | `streamlit run app.py` |
| Convert CLI | `python -m onnxruntime_genai.models.builder -m MODEL -o OUTPUT -p PRECISION -e PROVIDER` |
| Check model | Visit `https://huggingface.co/MODEL/blob/main/config.json` |
| Get help | Check `README.md` or `SUMMARY.md` |

---

## Resources

- 📖 Full Docs: `README.md`
- 📊 Comparison: `COMPARISON.md`
- 📝 Summary: `SUMMARY.md`
- 🔧 Test Script: `test_setup.py`
- 🚀 Setup Script: `setup.sh`
- 🐙 ONNX Runtime GenAI: https://github.com/microsoft/onnxruntime-genai
- 📚 Documentation: https://onnxruntime.ai/docs/genai/

---

**Ready to convert? Run `streamlit run app.py` and enter your Qwen3 model!** 🎉
