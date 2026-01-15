"""
Bias injector that applies biases to answers.
"""
from typing import Dict, Any, Optional
from loguru import logger

from .base import BaseBias
from .builtin_biases import BUILTIN_BIASES
from ..models.base import BaseModel
from ..models.registry import ModelRegistry


class BiasInjector:
    """Inject bias into text using rule-based or AI-based methods."""
    
    def __init__(
        self,
        bias_type: str,
        injector_type: str = "mock",
        model_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize bias injector.
        
        Args:
            bias_type: Type of bias to inject
            injector_type: "mock" (rule-based) or "openai"/"hf" (AI-based)
            model_config: Configuration for AI model (if using AI-based)
            prompt_template: Prompt template for AI-based injection
        """
        self.bias_type = bias_type
        self.injector_type = injector_type
        self.model_config = model_config or {}
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        
        # Initialize bias handler
        if bias_type in BUILTIN_BIASES:
            self.bias_handler = BUILTIN_BIASES[bias_type](config={})
        else:
            raise ValueError(f"Unknown bias type: {bias_type}. Available: {list(BUILTIN_BIASES.keys())}")
        
        # Initialize model if using AI-based injection
        self.model = None
        if injector_type != "mock":
            allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
            try:
                # Determine model name for registry
                model_name = (
                    self.model_config.get("model")
                    or self.model_config.get("model_name")
                    or self.model_config.get("model_id")
                    or "default"
                )
                self.model = ModelRegistry.create_model(
                    model_type=injector_type,
                    model_name=model_name,
                    config=self.model_config
                )
                if not self.model.is_available():
                    msg = "Bias injector model not available"
                    if allow_fallback:
                        logger.warning(f"{msg}, falling back to rule-based injection (allow_fallback_mock=True)")
                        self.injector_type = "mock"
                        self.model = None
                    else:
                        raise RuntimeError(
                            f"{msg}. Set bias.allow_fallback_mock: true to allow fallback to rule-based injection."
                        )
            except Exception as e:
                if allow_fallback:
                    logger.warning(f"Error initializing model for bias injection: {e}, falling back to rule-based")
                    self.injector_type = "mock"
                    self.model = None
                else:
                    raise
    
    def inject(self, text: str, question: Optional[str] = None) -> str:
        """
        Inject bias into text.
        
        Args:
            text: Text to inject bias into
            question: Optional question context
            
        Returns:
            Text with bias injected
        """
        if self.injector_type == "mock":
            # Rule-based injection
            context = {"question": question} if question else {}
            return self.bias_handler.apply(text, context)
        else:
            # AI-based injection
            if self.model is None:
                logger.error("BiasInjector.inject: model is None but injector_type != mock")
                raise RuntimeError("Bias injector model is not initialized.")
            
            # Build prompt
            if self.prompt_template:
                prompt = self.prompt_template.format(question=question or "", answer=text)
            else:
                prompt = f"Question: {question}\n\nOriginal Answer: {text}\n\nModify the answer to inject {self.bias_type} bias."
            
            try:
                system_prompt = self.system_prompt or f"You are a text modifier that injects {self.bias_type} bias into medical answers."
                biased_text = self.model.generate(prompt, system_prompt=system_prompt)
                return biased_text
            except Exception as e:
                allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
                logger.error(f"Error in AI-based bias injection: {e}")
                if allow_fallback:
                    logger.warning("Falling back to rule-based injection due to error (allow_fallback_mock=True)")
                    context = {"question": question} if question else {}
                    return self.bias_handler.apply(text, context)
                raise

