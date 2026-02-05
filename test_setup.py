#!/usr/bin/env python3
"""
Test script to verify ONNX Runtime GenAI builder is working correctly.
This checks if all required dependencies are installed and the builder can be imported.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    required_packages = [
        ("streamlit", "Streamlit"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("yaml", "PyYAML"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("onnx", "ONNX"),
    ]
    
    failed = []
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT INSTALLED")
            failed.append(display_name)
    
    # Test ONNX Runtime GenAI and its C-extension package `onnx_ir` which commonly fails to import
    try:
        try:
            import onnx_ir  # this is a compiled module inside onnxruntime-genai
            print("  ✓ onnx_ir (compiled GenAI runtime) imported")
        except Exception:
            print("  ✗ onnx_ir - NOT IMPORTABLE")
            # Try the pure-python package wrapper as a secondary check
            try:
                import importlib.util
                spec = importlib.util.find_spec("onnxruntime_genai")
                if spec is not None:
                    print("  ✓ onnxruntime_genai Python package found")
                    from onnxruntime_genai.models import builder
                    print("  ✓ ONNX Runtime GenAI Builder imported")
                    if hasattr(builder, 'create_model'):
                        print("  ✓ create_model function found")
                    else:
                        print("  ✗ create_model function NOT found")
                        failed.append("create_model function")
                else:
                    print("  ✗ onnxruntime_genai - NOT INSTALLED")
                    failed.append("ONNX Runtime GenAI")
            except ImportError as e:
                print(f"  ✗ ONNX Runtime GenAI Builder - {e}")
                failed.append("ONNX Runtime GenAI Builder")
    except Exception as e:
        print(f"  ✗ ONNX Runtime GenAI - Error: {e}")
        failed.append("ONNX Runtime GenAI")
    
    print()
    
    if failed:
        print("❌ Some packages are missing:")
        for pkg in failed:
            print(f"   - {pkg}")
        print()
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages are installed!")
        return True


def test_supported_models():
    """Test if we can access the supported models list."""
    print("\n🔍 Testing supported models...")
    
    try:
        from app_new import SUPPORTED_ARCHITECTURES
        print(f"  ✓ Found {len(SUPPORTED_ARCHITECTURES)} supported architectures")
        
        # Check for Qwen3
        if "Qwen3ForCausalLM" in SUPPORTED_ARCHITECTURES:
            print(f"  ✓ Qwen3 is supported!")
        else:
            print(f"  ✗ Qwen3 NOT in supported list")
            return False
        
        print()
        print("Supported architectures:")
        for arch, name in sorted(SUPPORTED_ARCHITECTURES.items()):
            print(f"  - {arch}: {name}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading supported models: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ONNX Converter - Dependency Check")
    print("=" * 60)
    print()
    
    imports_ok = test_imports()
    models_ok = test_supported_models()
    
    print()
    print("=" * 60)
    
    if imports_ok and models_ok:
        print("✅ All tests passed! You're ready to convert Qwen3 models!")
        print()
        print("To start the converter:")
        print("  streamlit run app.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
