"""
Google Gemini API client (optional dependency).
"""
import os
import time
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
        self.timeout = config.get("timeout", 120) if config else 120  # Default 120 seconds
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.retry_delay = config.get("retry_delay", 2) if config else 2  # Delay between retries in seconds

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
        timeout = kwargs.get("timeout", self.timeout)
        max_retries = kwargs.get("max_retries", self.max_retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        model = self._model
        if system_prompt:
            model = self._genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )

        # Retry logic for timeout and transient errors
        last_error = None
        for attempt in range(max_retries):
            try:
                # Import timeout exception
                from google.api_core import exceptions as google_exceptions
                
                # Note: google.generativeai may not support request_options directly
                # Timeout is handled at the gRPC level, we'll catch DeadlineExceeded exceptions
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_str = str(e)
                
                # Check if it's a quota error (429) - these usually need quota reset, not retry
                is_quota_error = False
                if "429" in error_str or "ResourceExhausted" in error_type:
                    if "quota" in error_str.lower() or "exceeded" in error_str.lower():
                        is_quota_error = True
                
                # Check if it's a retryable error (but not quota error)
                is_retryable = False
                if is_quota_error:
                    # Quota errors: return fallback immediately, don't retry
                    logger.error(f"Gemini API quota exceeded: {e}")
                    logger.error("Quota errors typically require waiting for quota reset (often 24 hours). Returning fallback response.")
                    return "[Response blocked - API quota exceeded. Please check your quota or wait for reset.]"
                elif "DeadlineExceeded" in error_type or "timeout" in error_str.lower():
                    is_retryable = True
                    logger.warning(f"Gemini API timeout (attempt {attempt + 1}/{max_retries}): {e}")
                elif "503" in error_str or "500" in error_str:
                    # Service unavailable or internal error (retryable, but not quota)
                    is_retryable = True
                    logger.warning(f"Gemini API transient error (attempt {attempt + 1}/{max_retries}): {e}")
                
                if is_retryable and attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Not retryable or out of retries
                    if "DeadlineExceeded" in error_type or "timeout" in error_str.lower():
                        logger.error(f"Gemini API timeout after {max_retries} attempts. Returning fallback response.")
                        return "[Response timeout - request took too long]"
                    else:
                        # Re-raise non-retryable errors
                        raise
        
        # If we exhausted retries, return fallback
        if last_error:
            error_str = str(last_error)
            if "DeadlineExceeded" in type(last_error).__name__ or "timeout" in error_str.lower():
                logger.error(f"Gemini API failed after {max_retries} attempts: {last_error}")
                return "[Response timeout - request took too long]"
            elif "429" in error_str or ("ResourceExhausted" in type(last_error).__name__ and "quota" in error_str.lower()):
                logger.error(f"Gemini API quota exceeded: {last_error}")
                return "[Response blocked - API quota exceeded. Please check your quota or wait for reset.]"
        
        # If we get here, response should be valid
        if 'response' not in locals():
            logger.error("Gemini API failed to generate response")
            return "[Response generation failed]"

        # Check for safety filter or other blocking reasons
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            
            # finish_reason values: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            # In the API, SAFETY is actually represented as finish_reason enum
            # Check if content was blocked
            if finish_reason and str(finish_reason) in ["SAFETY", "2"]:
                safety_ratings = getattr(candidate, "safety_ratings", None) or []
                safety_info = ", ".join([f"{r.category}:{r.probability}" for r in safety_ratings]) if safety_ratings else "unknown"
                logger.warning(f"Gemini safety filter blocked content. Reason: {finish_reason}, Safety ratings: {safety_info}")
                # Return a fallback response instead of raising error
                return "[Response blocked by safety filter]"
            
            if finish_reason and str(finish_reason) in ["MAX_TOKENS", "3"]:
                logger.warning(f"Gemini response truncated due to max tokens. Finish reason: {finish_reason}")
            
            # Try to get text from candidate
            content = getattr(candidate, "content", None)
            if content:
                parts = getattr(content, "parts", None) or []
                if parts and hasattr(parts[0], "text"):
                    return parts[0].text.strip()

        # Fallback: try response.text (may raise error if blocked)
        try:
            text = getattr(response, "text", None)
            if text:
                return text.strip()
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to access response.text: {e}. Using fallback parsing.")
        
        # If all else fails, return empty string
        logger.warning("Gemini response had no extractable text content")
        return ""

    def is_available(self) -> bool:
        if self._model is None:
            logger.debug("GeminiModel.is_available: False - client not initialized")
            return False
        if not self.api_key:
            logger.debug("GeminiModel.is_available: False - API key not found")
            return False
        return True
