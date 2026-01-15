"""
Anthropic Claude API client (optional dependency).
"""
import os
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class AnthropicModel(BaseModel):
    """Anthropic Claude model client."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)

        api_key_env_name = config.get("api_key_env", "ANTHROPIC_API_KEY") if config else "ANTHROPIC_API_KEY"
        self.api_key = os.getenv(api_key_env_name) if api_key_env_name else None
        if not self.api_key and config:
            self.api_key = config.get("api_key")

        if self.api_key:
            logger.info(f"AnthropicModel: API key found (env='{api_key_env_name}' or inline)")
        else:
            logger.warning(f"AnthropicModel: API key NOT found in environment variable '{api_key_env_name}'")

        self.temperature = config.get("temperature", 0.0) if config else 0.0
        self.max_tokens = config.get("max_tokens", 1000) if config else 1000

        self._anthropic = None
        self._client = None

        try:
            import anthropic
            self._anthropic = anthropic
            if self.api_key:
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("AnthropicModel: Using anthropic client")
        except ImportError:
            logger.error("AnthropicModel: anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"AnthropicModel: failed to initialize client: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if not self.is_available():
            raise RuntimeError("Anthropic client is not available. Check API key and package installation.")

        messages = [{"role": "user", "content": prompt}]
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)

        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or None,
            messages=messages,
        )

        content = getattr(response, "content", None)
        if isinstance(content, list):
            parts = []
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
            return "".join(parts).strip()
        return str(content).strip()

    def is_available(self) -> bool:
        if self._client is None:
            logger.debug("AnthropicModel.is_available: False - client not initialized")
            return False
        if not self.api_key:
            logger.debug("AnthropicModel.is_available: False - API key not found")
            return False
        return True
