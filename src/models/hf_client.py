"""
HuggingFace model client (optional dependency).
"""
import os
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
        except ImportError:
            logger.warning("transformers/torch not installed. HuggingFace client will not work.")
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        if self._transformers is None:
            raise RuntimeError("transformers package not installed")
        
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            self._tokenizer = self._transformers.AutoTokenizer.from_pretrained(self.model_name)
            self._model = self._transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(self._torch, self.dtype) if hasattr(self._torch, self.dtype) else self._torch.float32,
                device_map=self.device if self.device != "cpu" else None
            )
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

