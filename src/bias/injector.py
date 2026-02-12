"""
Bias injector that applies biases to answers.
"""
import re
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
        system_prompt: Optional[str] = None,
        injection_mode: str = "rewrite"
    ):
        """
        Initialize bias injector.
        
        Args:
            bias_type: Type of bias to inject
            injector_type: "mock" (rule-based) or "openai"/"vllm" (AI-based)
            model_config: Configuration for AI model (if using AI-based)
            prompt_template: Prompt template for AI-based injection
            injection_mode: "rewrite" for full rewrite, "word" for word/phrase replacement
        """
        self.bias_type = bias_type
        self.injector_type = injector_type
        self.model_config = model_config or {}
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        if injection_mode not in {"rewrite", "word"}:
            raise ValueError(f"Unknown injection_mode: {injection_mode}. Use 'rewrite' or 'word'.")
        self.injection_mode = injection_mode
        
        # Initialize bias handler (only needed for rule-based injection)
        if bias_type in BUILTIN_BIASES:
            self.bias_handler = BUILTIN_BIASES[bias_type](config={})
        else:
            self.bias_handler = None
        
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
                        if self.bias_handler is None:
                            raise RuntimeError(
                                f"{msg}. allow_fallback_mock=True but no rule-based bias handler exists "
                                f"for bias_type='{self.bias_type}'."
                            )
                        logger.warning(f"{msg}, falling back to rule-based injection (allow_fallback_mock=True)")
                        self.injector_type = "mock"
                        self.model = None
                    else:
                        raise RuntimeError(
                            f"{msg}. Set bias.allow_fallback_mock: true to allow fallback to rule-based injection."
                        )
            except Exception as e:
                if allow_fallback:
                    if self.bias_handler is None:
                        raise RuntimeError(
                            f"Error initializing model for bias injection: {e}. "
                            f"allow_fallback_mock=True but no rule-based bias handler exists for bias_type='{self.bias_type}'."
                        )
                    logger.warning(f"Error initializing model for bias injection: {e}, falling back to rule-based")
                    self.injector_type = "mock"
                    self.model = None
                else:
                    raise
    
    def _clean_biased_output(self, raw_output: str, original_answer: str) -> str:
        """
        Clean model output to extract only the modified answer.
        
        This method removes:
        - Labels like "Modified Answer:", "Answer:", etc.
        - Explanations or additional text
        - Quotation marks if they wrap the entire answer
        - Leading/trailing whitespace
        
        Args:
            raw_output: Raw output from the model
            original_answer: Original answer (for reference)
            
        Returns:
            Cleaned answer text
        """
        if not raw_output:
            return ""
        
        text = raw_output.strip()
        
        # Remove common labels/prefixes (case-insensitive)
        labels = [
            r"^(?:modified\s+)?answer\s*:?\s*",
            r"^(?:修改后的)?答案\s*:?\s*",
            r"^(?:modified\s+)?response\s*:?\s*",
            r"^(?:修改后的)?回答\s*:?\s*",
            r"^output\s*:?\s*",
            r"^输出\s*:?\s*",
        ]
        for label_pattern in labels:
            text = re.sub(label_pattern, "", text, flags=re.IGNORECASE)
            text = text.strip()
        
        # Remove quotation marks if they wrap the entire text
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        if (text.startswith('「') and text.endswith('」')) or (text.startswith('"') and text.endswith('"')):
            text = text[1:-1].strip()
        
        # If output contains "Example:" or similar, try to extract the answer part
        # Look for patterns like "Example:\n...\nModified Answer: ..."
        example_pattern = r"(?:example|示例)[\s\S]*?modified\s+answer\s*:?\s*(.+)"
        match = re.search(example_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
        
        # Remove trailing explanations that start with common phrases
        explanation_markers = [
            r"\n\s*(?:note|注意|说明|explanation|note:).*$",
            r"\n\s*(?:this|这|that|那).*$",
        ]
        for marker in explanation_markers:
            text = re.sub(marker, "", text, flags=re.IGNORECASE | re.DOTALL)
            text = text.strip()
        
        # If the cleaned text is too short (likely incomplete extraction), return original
        # But if it's significantly longer than original, it might be correct
        if len(text) < len(original_answer) * 0.3 and len(text) < 20:
            logger.warning(f"Cleaned output seems too short, using raw output. Original length: {len(original_answer)}, Cleaned: {len(text)}")
            return raw_output.strip()
        
        return text
    
    def inject(self, text: str, question: Optional[str] = None) -> str:
        """
        Inject bias into text.
        
        Args:
            text: Text to inject bias into (non-GT answer)
            question: Optional question context
            
        Returns:
            Text with bias injected (cleaned to ensure only modified answer is returned)
        """
        if self.injector_type == "mock":
            # Rule-based injection
            if self.bias_handler is None:
                raise ValueError(
                    f"Bias type '{self.bias_type}' does not have a rule-based handler. "
                    "Please use an AI-based injector (openai/vllm/gemini/anthropic) and define prompts.yaml."
                )
            context = {"question": question} if question else {}
            if self.injection_mode == "word":
                return self.bias_handler.apply_word_replacement(text, context)
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
                if self.injection_mode == "word":
                    system_prompt = (
                        system_prompt
                        + " Only replace words or short phrases; do not restructure, reorder, or add new sentences."
                    )
                raw_biased_text = self.model.generate(prompt, system_prompt=system_prompt)
                
                # Clean the output to ensure only the modified answer is returned
                cleaned_biased_text = self._clean_biased_output(raw_biased_text, text)
                
                # Log if significant cleaning was performed
                if len(raw_biased_text) != len(cleaned_biased_text) or raw_biased_text != cleaned_biased_text:
                    logger.debug(f"Cleaned bias injection output: {len(raw_biased_text)} -> {len(cleaned_biased_text)} chars")
                
                return cleaned_biased_text
            except Exception as e:
                allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
                logger.error(f"Error in AI-based bias injection: {e}")
                if allow_fallback:
                    if self.bias_handler is None:
                        raise RuntimeError(
                            f"Falling back requested but no rule-based bias handler exists for bias_type='{self.bias_type}'."
                        )
                    logger.warning("Falling back to rule-based injection due to error (allow_fallback_mock=True)")
                    context = {"question": question} if question else {}
                    return self.bias_handler.apply(text, context)
                raise

