"""
HuggingFace model client (optional dependency).
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class HuggingFaceModel(BaseModel):
    """HuggingFace model client."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.device = config.get("device", "cpu") if config else "cpu"
        self.dtype = config.get("dtype", "float32") if config else "float32"
        self.max_length = config.get("max_length", 2048) if config else 2048
        self.use_modelscope = config.get("use_modelscope", False) if config else False
        
        # Local model cache directory (default: src/models/)
        self.local_cache_dir = config.get("local_cache_dir") if config else None
        if self.local_cache_dir:
            cache_path = Path(self.local_cache_dir)
            # If relative path, resolve relative to project root
            if not cache_path.is_absolute():
                # Get project root (assuming this file is in src/models/)
                project_root = Path(__file__).parent.parent.parent
                cache_path = project_root / cache_path
            self.local_cache_dir = cache_path
            self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import transformers, but don't fail if not available
        self._transformers = None
        self._torch = None
        self._tokenizer = None
        self._model = None
        
        try:
            import transformers
            import torch
            self._transformers = transformers
            self._torch = torch
            
            # Check device availability
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Load model (lazy loading - only when first used)
            logger.info(f"HuggingFace client initialized for {model_name} (device: {self.device})")
            if self.local_cache_dir:
                logger.info(f"Local cache directory: {self.local_cache_dir}")
        except ImportError:
            logger.warning("transformers/torch not installed. HuggingFace client will not work.")
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        if self._transformers is None:
            raise RuntimeError("transformers package not installed")
        
        try:
            # Determine if we should use local cache
            local_model_path = None
            if self.local_cache_dir:
                # Create a safe directory name from model_name (replace / with _)
                safe_model_name = self.model_name.replace("/", "_")
                local_model_path = self.local_cache_dir / safe_model_name
                
                # Check if model exists locally
                if local_model_path.exists() and any(local_model_path.iterdir()):
                    logger.info(f"Found local model at {local_model_path}, using cached version")
                    model_path = str(local_model_path)
                else:
                    logger.info(f"Model not found locally. Will download to {local_model_path}")
                    model_path = self.model_name
            else:
                model_path = self.model_name
            
            logger.info(f"Loading HuggingFace model: {model_path}")
            
            # Load tokenizer and model
            if local_model_path and local_model_path.exists() and any(local_model_path.iterdir()):
                # Load from local cache
                self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(str(local_model_path))
                self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                    str(local_model_path),
                    torch_dtype=getattr(self._torch, self.dtype) if hasattr(self._torch, self.dtype) else self._torch.float32,
                    device_map=self.device if self.device != "cpu" else None
                )
            else:
                # Download from HuggingFace or ModelScope
                # Check if model is from ModelScope (modelscope.cn) or use ModelScope mirror
                use_modelscope = config.get("use_modelscope", False) if config else False
                if use_modelscope or "modelscope" in self.model_name.lower():
                    logger.info(f"Downloading model {self.model_name} from ModelScope")
                    # Try to use modelscope library if available
                    try:
                        from modelscope import snapshot_download
                        model_dir = snapshot_download(self.model_name, cache_dir=str(self.local_cache_dir.parent) if self.local_cache_dir else None)
                        model_path = model_dir
                        logger.info(f"Downloaded from ModelScope to {model_path}")
                    except ImportError:
                        logger.warning("modelscope library not installed. Trying transformers with ModelScope mirror...")
                        # Fallback: use transformers with ModelScope mirror via environment variable
                        import os
                        original_endpoint = os.environ.get("HF_ENDPOINT")
                        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # Use HuggingFace mirror that supports ModelScope
                        try:
                            model_path = self.model_name
                        finally:
                            if original_endpoint:
                                os.environ["HF_ENDPOINT"] = original_endpoint
                            elif "HF_ENDPOINT" in os.environ:
                                del os.environ["HF_ENDPOINT"]
                else:
                    logger.info(f"Downloading model {self.model_name} from HuggingFace")
                    model_path = self.model_name
                
                self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(model_path)
                self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=getattr(self._torch, self.dtype) if hasattr(self._torch, self.dtype) else self._torch.float32,
                    device_map=self.device if self.device != "cpu" else None
                )
                
                # Save to local cache if configured
                if local_model_path:
                    logger.info(f"Saving model to {local_model_path}")
                    local_model_path.mkdir(parents=True, exist_ok=True)
                    self._tokenizer.save_pretrained(str(local_model_path))
                    self._model.save_pretrained(str(local_model_path))
                    logger.info(f"Model saved to {local_model_path}")
            
            if self.device == "cpu":
                self._model = self._model.to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using HuggingFace model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if not self.is_available():
            raise RuntimeError("HuggingFace client is not available. Check package installation.")
        
        self._load_model()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            inputs = self._tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self._torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 512),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=kwargs.get("do_sample", True),
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            if full_prompt in generated_text:
                generated_text = generated_text[len(full_prompt):].strip()
            
            return generated_text
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if HuggingFace client is available."""
        return self._transformers is not None

