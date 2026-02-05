"""Convert Hugging Face models to ONNX format using ONNX Runtime GenAI builder.

This application provides a Streamlit interface for converting Hugging Face models
to ONNX format using the ONNX Runtime GenAI builder. It supports:
- Modern models including Qwen3, Gemma3, Phi4, SmolLM3, and more
- Multiple precision options (fp32, fp16, bf16, int4)
- Multiple execution providers (CPU, CUDA, DML, WebGPU)
- Advanced quantization options
- README generation with metadata
- Upload to Hugging Face Hub
"""

import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import yaml
from huggingface_hub import HfApi, hf_hub_download, model_info, whoami

# Import the ONNX Runtime GenAI builder
try:
    # Attempt to import the builder module
    # This assumes the builder.py and related files are available
    import importlib.util
    
    # Try to find builder.py - check if it exists locally or in installed package
    builder_spec = importlib.util.find_spec("onnxruntime_genai.models.builder")
    if builder_spec is None:
        raise ImportError(
            "Could not find onnxruntime-genai builder. "
            "Please install with: pip install onnxruntime-genai"
        )
    
    builder_module = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder_module)
    create_model = builder_module.create_model
    parse_extra_options = builder_module.parse_extra_options
except ImportError as e:
    st.error(
        f"Failed to import ONNX Runtime GenAI builder: {e}\n"
        "Please ensure onnxruntime-genai is installed: pip install onnxruntime-genai"
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported model architectures in ONNX Runtime GenAI
SUPPORTED_ARCHITECTURES = {
    "ChatGLMForConditionalGeneration": "ChatGLM",
    "ChatGLMModel": "ChatGLM",
    "Ernie4_5_ForCausalLM": "Ernie",
    "GemmaForCausalLM": "Gemma",
    "Gemma2ForCausalLM": "Gemma 2",
    "Gemma3ForCausalLM": "Gemma 3",
    "Gemma3ForConditionalGeneration": "Gemma 3 (multimodal)",
    "GptOssForCausalLM": "GPT-OSS",
    "GraniteForCausalLM": "Granite",
    "LlamaForCausalLM": "Llama",
    "MistralForCausalLM": "Mistral",
    "NemotronForCausalLM": "Nemotron",
    "OlmoForCausalLM": "OLMo",
    "PhiForCausalLM": "Phi",
    "Phi3ForCausalLM": "Phi-3",
    "PhiMoEForCausalLM": "Phi-3 MoE",
    "Phi3SmallForCausalLM": "Phi-3 Small",
    "Phi3VForCausalLM": "Phi-3 Vision",
    "Phi4MMForCausalLM": "Phi-4",
    "Qwen2ForCausalLM": "Qwen2",
    "Qwen3ForCausalLM": "Qwen3",
    "SmolLM3ForCausalLM": "SmolLM3",
    "Qwen2_5_VLForConditionalGeneration": "Qwen2.5-VL",
}


@dataclass
class Config:
    """Application configuration containing authentication and path settings.

    Attributes:
        hf_token: Hugging Face API token (user token takes precedence over system token)
        hf_username: Hugging Face username associated with the token
        is_using_user_token: True if using a user-provided token, False if using system token
        hf_base_url: Base URL for Hugging Face Hub
    """

    hf_token: str
    hf_username: str
    is_using_user_token: bool
    hf_base_url: str = "https://huggingface.co"

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables and Streamlit session state.

        Priority order for tokens:
        1. User-provided token from Streamlit session (st.session_state.user_hf_token)
        2. System token from environment variable (HF_TOKEN)

        Returns:
            Config: Initialized configuration object

        Raises:
            ValueError: If no valid token is available
        """
        system_token = os.getenv("HF_TOKEN")
        user_token = st.session_state.get("user_hf_token")

        # Determine username based on which token is being used
        if user_token:
            hf_username = whoami(token=user_token)["name"]
        else:
            hf_username = (
                os.getenv("SPACE_AUTHOR_NAME") or whoami(token=system_token)["name"]
            )

        # User token takes precedence over system token
        hf_token = user_token or system_token

        if not hf_token:
            raise ValueError(
                "When the user token is not provided, the system token must be set."
            )

        return cls(
            hf_token=hf_token,
            hf_username=hf_username,
            is_using_user_token=bool(user_token),
        )


class ModelConverter:
    """Handles model conversion to ONNX format using ONNX Runtime GenAI builder.

    This class manages the entire conversion workflow:
    1. Fetching original model metadata and README
    2. Running the ONNX conversion using the builder
    3. Generating an enhanced README with merged metadata
    4. Uploading the converted model to Hugging Face Hub

    Attributes:
        config: Application configuration containing tokens and paths
        api: Hugging Face API client for repository operations
    """

    def __init__(self, config: Config):
        """Initialize the converter with configuration.

        Args:
            config: Application configuration object
        """
        self.config = config
        self.api = HfApi(token=config.hf_token)

    # ============================================================================
    # README Processing Methods
    # ============================================================================

    def _fetch_original_readme(self, repo_id: str) -> str:
        """Download the README from the original model repository.

        Args:
            repo_id: Hugging Face model repository ID (e.g., 'username/model-name')

        Returns:
            str: Content of the README file, or empty string if not found
        """
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id, filename="README.md", token=self.config.hf_token
            )
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            # Silently fail if README doesn't exist or can't be downloaded
            return ""

    def _strip_yaml_frontmatter(self, text: str) -> str:
        """Remove YAML frontmatter from text, returning only the body.

        YAML frontmatter is delimited by '---' at the start and end.

        Args:
            text: Text that may contain YAML frontmatter

        Returns:
            str: Text with frontmatter removed, or original text if no frontmatter found
        """
        if not text:
            return ""
        if text.startswith("---"):
            match = re.match(r"^---[\s\S]*?\n---\s*\n", text)
            if match:
                return text[match.end() :]
        return text

    def _extract_yaml_frontmatter(self, text: str) -> Tuple[dict, str]:
        """Parse and extract YAML frontmatter from text.

        Args:
            text: Text that may contain YAML frontmatter

        Returns:
            Tuple containing:
            - dict: Parsed YAML frontmatter as a dictionary (empty dict if none found)
            - str: Remaining body text after the frontmatter
        """
        if not text or not text.startswith("---"):
            return {}, text or ""

        # Match YAML frontmatter pattern: ---\n...content...\n---\n
        match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n", text)
        if not match:
            return {}, text

        frontmatter_text = match.group(1)
        body = text[match.end() :]

        # Parse YAML safely, returning empty dict on any error
        try:
            parsed_data = yaml.safe_load(frontmatter_text)
            if not isinstance(parsed_data, dict):
                parsed_data = {}
        except Exception:
            parsed_data = {}

        return parsed_data, body

    # ============================================================================
    # Model Conversion Methods
    # ============================================================================

    def check_model_compatibility(self, model_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if a model is compatible with ONNX Runtime GenAI builder.

        Args:
            model_id: Hugging Face model repository ID

        Returns:
            Tuple containing:
            - bool: True if model is compatible, False otherwise
            - Optional[str]: Architecture name if found
            - Optional[str]: Error message if incompatible
        """
        try:
            info = model_info(repo_id=model_id, token=self.config.hf_token)
            config = info.config
            
            if not config or "architectures" not in config:
                return False, None, "Model config does not specify architecture"
            
            architectures = config.get("architectures", [])
            if not architectures:
                return False, None, "No architectures found in model config"
            
            architecture = architectures[0]
            if architecture not in SUPPORTED_ARCHITECTURES:
                return False, architecture, f"Architecture '{architecture}' is not supported. Supported architectures: {', '.join(SUPPORTED_ARCHITECTURES.keys())}"
            
            return True, architecture, None
            
        except Exception as e:
            return False, None, f"Error checking model compatibility: {str(e)}"

    def convert_model(
        self,
        input_model_id: str,
        output_dir: str,
        precision: str = "fp16",
        execution_provider: str = "cuda",
        cache_dir: str = "./cache_dir",
        extra_options: Optional[dict] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Convert a Hugging Face model to ONNX format using ONNX Runtime GenAI builder.

        Args:
            input_model_id: Hugging Face model repository ID
            output_dir: Directory where ONNX model will be saved
            precision: Model precision (fp32, fp16, bf16, int4)
            execution_provider: Target execution provider (cpu, cuda, dml, webgpu)
            cache_dir: Cache directory for Hugging Face files
            extra_options: Additional options for the builder

        Returns:
            Tuple containing:
            - bool: True if conversion succeeded, False otherwise
            - Optional[str]: Error message if failed, success message if succeeded
        """
        try:
            # Create output and cache directories
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            # Set HF_TOKEN environment variable for the builder
            env = os.environ.copy()
            env["HF_TOKEN"] = self.config.hf_token

            # Prepare extra options
            if extra_options is None:
                extra_options = {}
            
            # Add hf_token to extra_options
            extra_options["hf_token"] = self.config.hf_token

            # Call the builder's create_model function
            logger.info(f"Converting {input_model_id} to ONNX...")
            logger.info(f"Precision: {precision}, Execution Provider: {execution_provider}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Extra options: {extra_options}")

            create_model(
                model_name=input_model_id,
                input_path="",  # Empty means download from HF
                output_dir=output_dir,
                precision=precision,
                execution_provider=execution_provider,
                cache_dir=cache_dir,
                **extra_options,
            )

            return True, "Conversion successful!"

        except Exception as e:
            logger.exception("Conversion failed")
            return False, str(e)

    # ============================================================================
    # Upload Methods
    # ============================================================================

    def upload_model(self, local_dir: str, output_model_id: str) -> Optional[str]:
        """Upload the converted ONNX model to Hugging Face Hub.

        This method:
        1. Creates the target repository (if it doesn't exist)
        2. Uploads all model files to the repository

        Args:
            local_dir: Local directory containing the ONNX model
            output_model_id: Target repository ID for the ONNX model

        Returns:
            Optional[str]: Error message if upload failed, None if successful
        """
        try:
            # Create the target repository (public by default)
            self.api.create_repo(output_model_id, exist_ok=True, private=False)

            # Upload all files from the local directory
            self.api.upload_folder(
                folder_path=local_dir, 
                repo_id=output_model_id,
                commit_message="Upload ONNX model converted with ONNX Runtime GenAI"
            )

            return None  # Success

        except Exception as e:
            return str(e)

    # ============================================================================
    # README Generation Methods
    # ============================================================================

    def generate_readme(self, input_model_id: str, precision: str, execution_provider: str, output_model_id: Optional[str] = None) -> str:
        """Generate an enhanced README for the ONNX model.

        This method creates a README that:
        1. Merges metadata from the original model with ONNX-specific metadata
        2. Adds a description and usage instructions
        3. Appends the original model's README content

        Args:
            input_model_id: Original model repository ID
            precision: Model precision used for conversion
            execution_provider: Execution provider used for conversion

        Returns:
            str: Complete README content in Markdown format with YAML frontmatter
        """
        # Fetch pipeline tag from model metadata (if available)
        try:
            info = model_info(repo_id=input_model_id, token=self.config.hf_token)
            pipeline_tag = getattr(info, "pipeline_tag", None)
        except Exception:
            pipeline_tag = None

        # Fetch and parse the original README
        original_text = self._fetch_original_readme(input_model_id)
        original_meta, original_body = self._extract_yaml_frontmatter(original_text)
        original_body = (
            original_body or self._strip_yaml_frontmatter(original_text)
        ).strip()

        # Merge original metadata with ONNX-specific metadata
        merged_meta = {}
        if isinstance(original_meta, dict):
            merged_meta.update(original_meta)
        merged_meta["library_name"] = "onnxruntime"
        merged_meta["base_model"] = [input_model_id]
        merged_meta["tags"] = merged_meta.get("tags", []) + ["onnx", "onnxruntime-genai"]
        if pipeline_tag is not None:
            merged_meta["pipeline_tag"] = pipeline_tag

        # Generate YAML frontmatter
        frontmatter_yaml = yaml.safe_dump(merged_meta, sort_keys=False).strip()
        header = f"---\n{frontmatter_yaml}\n---\n\n"

        # Build README sections
        readme_sections: List[str] = []
        readme_sections.append(header)

        # Add title
        model_name = input_model_id.split("/")[-1]
        readme_sections.append(f"# {model_name} (ONNX)\n")

        # Add description
        readme_sections.append(
            f"This is an ONNX version of [{input_model_id}](https://huggingface.co/{input_model_id}). "
            "It was converted using the ONNX Runtime GenAI builder.\n"
        )

        # Add conversion details
        readme_sections.append("## Conversion Details\n")
        readme_sections.append(f"- **Precision**: {precision}")
        readme_sections.append(f"- **Execution Provider**: {execution_provider}")
        readme_sections.append(f"- **Base Model**: [{input_model_id}](https://huggingface.co/{input_model_id})")

        # Add usage section
        readme_sections.append("\n## Usage with ONNX Runtime GenAI\n")
        readme_sections.append("```python")
        readme_sections.append("import onnxruntime_genai as og")
        readme_sections.append("")
        readme_sections.append("# Load the model")
        # If an explicit output_model_id was not provided, derive a sensible default
        if not output_model_id:
            model_name = input_model_id.split("/")[-1]
            derived_owner = self.config.hf_username
            output_model_id = f"{derived_owner}/{model_name}-ONNX"

        readme_sections.append(f'model = og.Model("{output_model_id}")')
        readme_sections.append("tokenizer = og.Tokenizer(model)")
        readme_sections.append("")
        readme_sections.append("# Generate text")
        readme_sections.append('prompt = "Your prompt here"')
        readme_sections.append("tokens = tokenizer.encode(prompt)")
        readme_sections.append("params = og.GeneratorParams(model)")
        readme_sections.append("params.set_search_options(max_length=200)")
        readme_sections.append("params.input_ids = tokens")
        readme_sections.append("")
        readme_sections.append("generator = og.Generator(model, params)")
        readme_sections.append("while not generator.is_done():")
        readme_sections.append("    generator.compute_logits()")
        readme_sections.append("    generator.generate_next_token()")
        readme_sections.append("")
        readme_sections.append("output_tokens = generator.get_sequence(0)")
        readme_sections.append("text = tokenizer.decode(output_tokens)")
        readme_sections.append("print(text)")
        readme_sections.append("```")

        # Append original README content (if available)
        if original_body:
            readme_sections.append("\n---\n")
            readme_sections.append("# Original Model Card\n")
            readme_sections.append(original_body)

        return "\n\n".join(readme_sections) + "\n"


def main():
    """Main application entry point for the Streamlit interface.

    This function:
    1. Initializes configuration and converter
    2. Displays the UI for model input and options
    3. Handles the conversion workflow
    4. Shows progress and results to the user
    """
    st.write("## Convert a Hugging Face model to ONNX using ONNX Runtime GenAI")

    try:
        # Initialize configuration and converter
        config = Config.from_env()
        converter = ModelConverter(config)

        # Get model ID from user
        input_model_id = st.text_input(
            "Enter the Hugging Face model ID to convert",
            placeholder="e.g., Qwen/Qwen3-0.5B-Instruct",
            help="Enter the full model ID from Hugging Face (e.g., 'username/model-name')"
        )

        if not input_model_id:
            st.info("👆 Enter a model ID to get started")
            return

        # Check model compatibility
        with st.spinner("Checking model compatibility..."):
            is_compatible, architecture, error_msg = converter.check_model_compatibility(input_model_id)
            
            if not is_compatible:
                st.error(f"❌ Model is not compatible: {error_msg}")
                if architecture:
                    st.write(f"Detected architecture: `{architecture}`")
                st.write("### Supported architectures:")
                for arch, name in SUPPORTED_ARCHITECTURES.items():
                    st.write(f"- `{arch}` ({name})")
                return
            
            st.success(f"✅ Model is compatible! Architecture: {SUPPORTED_ARCHITECTURES[architecture]}")

        # Optional: User token input
        st.text_input(
            "Optional: Your Hugging Face write token",
            type="password",
            key="user_hf_token",
            help="Fill this if you want to upload the model under your account. Leave empty to use system token."
        )

        # Conversion options
        st.write("### Conversion Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            precision = st.selectbox(
                "Precision",
                options=["fp16", "fp32", "bf16", "int4"],
                index=0,
                help="Model precision. fp16 is recommended for most cases. int4 for quantized models."
            )
        
        with col2:
            execution_provider = st.selectbox(
                "Execution Provider",
                options=["cuda", "cpu", "dml", "webgpu"],
                index=0,
                help="Target execution provider. CUDA for NVIDIA GPUs, CPU for CPU inference."
            )

        # Advanced options
        with st.expander("Advanced Options"):
            st.write("Configure additional conversion options:")
            
            # INT4 quantization options (only show if precision is int4)
            if precision == "int4":
                int4_block_size = st.selectbox(
                    "INT4 Block Size",
                    options=[16, 32, 64, 128, 256],
                    index=1,  # Default to 32
                    help="Block size for INT4 quantization"
                )
                
                int4_is_symmetric = st.checkbox(
                    "INT4 Symmetric Quantization",
                    value=True,
                    help="Use symmetric quantization (int4) vs asymmetric (uint4)"
                )
                
                int4_accuracy_level = st.selectbox(
                    "INT4 Accuracy Level",
                    options=[0, 1, 2, 3, 4],
                    index=4,  # Default to 4 (int8)
                    help="Minimum accuracy level for activation. 4=int8, 3=bf16, 2=fp16, 1=fp32, 0=no constraint"
                )
            
            # General options
            exclude_embeds = st.checkbox(
                "Exclude Embedding Layer",
                value=False,
                help="Remove embedding layer from ONNX model (use inputs_embeds instead of input_ids)"
            )
            
            exclude_lm_head = st.checkbox(
                "Exclude LM Head",
                value=False,
                help="Remove language modeling head (output hidden_states instead of logits)"
            )
            
            if execution_provider == "cuda":
                enable_cuda_graph = st.checkbox(
                    "Enable CUDA Graph",
                    value=False,
                    help="Enable CUDA graph capture for faster inference"
                )

        # Determine output repository
        if config.hf_username == input_model_id.split("/")[0]:
            same_repo = st.checkbox(
                "Upload to the existing repository",
                help="Upload ONNX weights to the same repository as the original model"
            )
        else:
            same_repo = False

        model_name = input_model_id.split("/")[-1]
        output_model_id = f"{config.hf_username}/{model_name}"

        # Add -ONNX suffix if creating a new repository
        if not same_repo:
            output_model_id += "-ONNX"

        output_model_url = f"{config.hf_base_url}/{output_model_id}"

        # Check if model already exists
        if not same_repo and converter.api.repo_exists(output_model_id):
            st.warning("⚠️ This model has already been converted!")
            st.link_button(f"Go to {output_model_id}", output_model_url, type="primary")
            
            if st.button("Convert anyway (overwrite)", type="secondary"):
                pass  # Continue with conversion
            else:
                return

        # Show where the model will be uploaded
        st.write("### Output Repository")
        st.code(output_model_url, language="plaintext")

        # Wait for user confirmation before proceeding
        if not st.button(label="Start Conversion", type="primary"):
            return

        # Prepare extra options
        extra_options = {}
        
        if precision == "int4":
            extra_options["int4_block_size"] = int4_block_size
            extra_options["int4_is_symmetric"] = int4_is_symmetric
            if int4_accuracy_level > 0:
                extra_options["int4_accuracy_level"] = int4_accuracy_level
        
        if exclude_embeds:
            extra_options["exclude_embeds"] = True
        
        if exclude_lm_head:
            extra_options["exclude_lm_head"] = True
        
        if execution_provider == "cuda" and "enable_cuda_graph" in locals() and enable_cuda_graph:
            extra_options["enable_cuda_graph"] = True

        # Create temporary output directory
        temp_output_dir = f"./onnx_models/{input_model_id.replace('/', '_')}"

        # Step 1: Convert the model to ONNX
        with st.spinner("Converting model to ONNX... This may take several minutes."):
            success, message = converter.convert_model(
                input_model_id=input_model_id,
                output_dir=temp_output_dir,
                precision=precision,
                execution_provider=execution_provider,
                extra_options=extra_options,
            )
            
            if not success:
                st.error(f"❌ Conversion failed: {message}")
                return

            st.success("✅ Conversion successful!")
            st.code(message)

        # Step 2: Generate README
        with st.spinner("Generating README..."):
            # Pass the final output_model_id so the README references the correct repo
            readme_content = converter.generate_readme(
                input_model_id=input_model_id,
                precision=precision,
                execution_provider=execution_provider,
                output_model_id=output_model_id,
            )
            
            # Write README to output directory
            readme_path = Path(temp_output_dir) / "README.md"
            readme_path.write_text(readme_content, encoding="utf-8")
            
            st.success("✅ README generated!")

            # Prepare a downloadable zip of the ONNX model in case upload fails
            zip_path = f"{temp_output_dir}.zip"
            try:
                # Remove any existing zip to avoid stale files
                if os.path.exists(zip_path):
                    os.remove(zip_path)

                # Create zip archive of the output directory
                shutil.make_archive(base_name=temp_output_dir, format="zip", root_dir=temp_output_dir)

                # Offer a download button (reads zip into memory briefly)
                try:
                    with open(zip_path, "rb") as zf:
                        zip_bytes = zf.read()
                    st.download_button(
                        label="Download ONNX model (zip)",
                        data=zip_bytes,
                        file_name=f"{model_name}-onnx.zip",
                        mime="application/zip",
                    )
                except Exception:
                    logger.exception("Failed to read zip for download button")
                    st.warning("Download unavailable: could not prepare archive for download.")

            except Exception as e:
                logger.exception("Failed to create download archive")
                st.warning(f"Could not create download package: {e}")

        # Step 3: Upload the converted model to Hugging Face
        with st.spinner("Uploading model to Hugging Face..."):
            error = converter.upload_model(temp_output_dir, output_model_id)
            if error:
                st.error(f"❌ Upload failed: {error}")
                return

            st.success("✅ Upload successful!")
            st.balloons()
            st.write("### 🎉 Conversion Complete!")
            st.write("You can now use your ONNX model on Hugging Face!")
            st.link_button(f"View {output_model_id}", output_model_url, type="primary")

        # Clean up temporary directory
        try:
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        except Exception:
            pass

    except Exception as e:
        logger.exception("Application error")
        st.error(f"❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
