"""
Google Gemini API client (optional dependency).
"""
import os
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class GeminiModel(BaseModel):
    """Google Gemini model client."""

    def __init__(self, model_name: str = "gemini-1.5-pro", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)

        api_key_env_name = config.get("api_key_env", "GEMINI_API_KEY") if config else "GEMINI_API_KEY"
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        if not self.api_key and config:
            self.api_key = config.get("api_key")
        if not self.api_key:
            # Common alternative env var used by Google AI Studio
            self.api_key = os.getenv("GOOGLE_API_KEY")

        if self.api_key:
            logger.info(f"GeminiModel: API key found (env='{api_key_env_name}' or inline)")
        else:
            logger.warning(f"GeminiModel: API key NOT found in environment variable '{api_key_env_name}'")

        self.temperature = config.get("temperature", 0.0) if config else 0.0
        self.max_tokens = config.get("max_tokens", 1000) if config else 1000

        self._genai = None
        self._model = None

        try:
            import google.generativeai as genai
            self._genai = genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=None,
                )
                logger.info("GeminiModel: Using google.generativeai client")
        except ImportError:
            logger.error("GeminiModel: google-generativeai package not installed. Install with: pip install google-generativeai")
        except Exception as e:
            logger.error(f"GeminiModel: failed to initialize client: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Gemini client is not available. Check API key and package installation.")

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        model = self._model
        if system_prompt:
            model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        # Fallback to candidate parsing
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            if parts and hasattr(parts[0], "text"):
                return parts[0].text.strip()
        return ""

    def is_available(self) -> bool:
        if self._model is None:
            logger.debug("GeminiModel.is_available: False - client not initialized")
            return False
        if not self.api_key:
            logger.debug("GeminiModel.is_available: False - API key not found")
            return False
        return True
