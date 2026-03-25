"""
OpenAI API client (optional dependency).
"""
import os
import sys
import importlib.metadata
import site
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API model client."""
    
    def __init__(self, model_name: str = "gpt-4", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        
        # OpenAI-compatible configuration
        api_key_env_name = config.get("api_key_env", "OPENAI_API_KEY") if config else "OPENAI_API_KEY"
        api_key = None
        if api_key_env_name:
            api_key = os.getenv(api_key_env_name)
        if not api_key and config:
            api_key = config.get("api_key")
        self.api_key = api_key
        self.allow_empty_api_key = bool(config.get("allow_empty_api_key", False)) if config else False
        self.base_url = (config.get("base_url") or config.get("api_base")) if config else None
        if not self.base_url and config:
            base_url_env_name = config.get("base_url_env")
            if base_url_env_name:
                self.base_url = os.getenv(str(base_url_env_name))
        self.organization = config.get("organization") if config else None
        self.project = config.get("project") if config else None
        self.default_headers = config.get("default_headers") if config else None
        # Some OpenAI-compatible servers (e.g., certain vLLM chat templates)
        # require strict user/assistant alternation and reject a leading
        # "system" role. For base_url mode we allow flattening system text
        # into a single user message by default.
        self.flatten_system_for_base_url = (
            bool(config.get("flatten_system_for_base_url", True)) if config else True
        )

        if self.api_key:
            logger.info(f"OpenAIModel: API key found (env='{api_key_env_name}' or inline)")
        elif self.allow_empty_api_key:
            logger.info("OpenAIModel: API key missing but allow_empty_api_key=True")
        else:
            logger.warning(f"OpenAIModel: API key NOT found in environment variable '{api_key_env_name}'")

        if self.base_url:
            logger.info(f"OpenAIModel: base_url set to {self.base_url}")
        
        self.temperature = config.get("temperature", 0.0) if config else 0.0
        # Backward compatible: config may specify max_tokens (legacy) or max_completion_tokens (newer models).
        self.max_tokens = config.get("max_tokens", 1000) if config else 1000
        self.max_completion_tokens = config.get("max_completion_tokens", None) if config else None
        
        # Import openai (we require the >=1.0 SDK).
        self._openai = None
        self._openai_client = None  # for >=1.0

        def _log_openai_info(openai_mod, label: str) -> None:
            try:
                v = importlib.metadata.version("openai")
            except Exception:
                v = getattr(openai_mod, "__version__", "unknown")
            p = getattr(openai_mod, "__file__", "unknown")
            logger.info(f"OpenAIModel[{label}]: python={sys.executable}")
            logger.info(f"OpenAIModel[{label}]: openai version={v}, module_path={p}")
            logger.info(f"OpenAIModel[{label}]: openai_has_OpenAI={hasattr(openai_mod, 'OpenAI')}")

        def _remove_user_site_from_syspath() -> bool:
            """Remove user-site from sys.path to avoid shadowing conda/venv packages."""
            try:
                user_site = site.getusersitepackages()
            except Exception:
                return False
            if not user_site:
                return False
            removed = False
            new_path = []
            for p in list(sys.path):
                if isinstance(p, str) and (p == user_site or p.startswith(user_site + os.sep)):
                    removed = True
                    continue
                new_path.append(p)
            if removed:
                sys.path[:] = new_path
            return removed

        def _purge_openai_modules() -> None:
            for k in list(sys.modules.keys()):
                if k == "openai" or k.startswith("openai."):
                    sys.modules.pop(k, None)

        try:
            import openai
            self._openai = openai
            _log_openai_info(openai, "initial")

            if not hasattr(openai, "OpenAI"):
                # Often caused by user-site openai==0.x shadowing a proper conda/venv install.
                removed = _remove_user_site_from_syspath()
                if removed:
                    logger.warning("OpenAIModel: detected legacy openai SDK; removed user-site packages from sys.path and retrying import")
                    _purge_openai_modules()
                    import openai as openai_retry  # type: ignore
                    self._openai = openai_retry
                    _log_openai_info(openai_retry, "after_user_site_removed")

            if not hasattr(self._openai, "OpenAI"):
                raise RuntimeError(
                    "OpenAIModel requires openai>=1.0.0 (new SDK with OpenAI() client). "
                    "Your current Python is importing an older openai package (likely from user-site). "
                    "Recommended fixes:\n"
                    "  1) Ensure you run the project in the intended conda/venv (check sys.executable)\n"
                    "  2) Uninstall the user-site openai (common culprit):\n"
                    "     python -m pip uninstall -y openai\n"
                    "  3) Install/upgrade openai>=1.0.0 in the SAME environment used to run experiments:\n"
                    "     pip install -U openai>=1.0.0\n"
                    "You can verify with:\n"
                    "  python -c \"import sys,openai; print(sys.executable); print(getattr(openai,'__version__','?')); print(openai.__file__)\""
                )

            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key or ("EMPTY" if self.allow_empty_api_key else None),
            }
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.organization:
                client_kwargs["organization"] = self.organization
            if self.project:
                client_kwargs["project"] = self.project
            if isinstance(self.default_headers, dict) and self.default_headers:
                client_kwargs["default_headers"] = self.default_headers

            if client_kwargs.get("api_key") is None:
                raise RuntimeError("OpenAIModel: API key is required but missing.")

            self._openai_client = self._openai.OpenAI(**client_kwargs)
            logger.info("OpenAIModel: Using openai>=1.0 client")
            if not self.api_key and not self.allow_empty_api_key:
                logger.warning(f"OpenAIModel: OpenAI package imported but no API key available")
        except ImportError:
            logger.error("OpenAIModel: openai package not installed. Install with: pip install openai>=1.0.0")
        except Exception as e:
            logger.error(f"OpenAIModel: failed to initialize OpenAI client: {e}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client is not available. Check API key and package installation.")
        
        messages = []
        used_flattened_system = False
        if system_prompt:
            should_flatten_system = bool(
                self.base_url and self.flatten_system_for_base_url
            )
            if should_flatten_system:
                used_flattened_system = True
                merged_prompt = (
                    f"[System Instruction]\n{system_prompt}\n\n"
                    f"[User Question]\n{prompt}"
                )
                messages.append({"role": "user", "content": merged_prompt})
            else:
                messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
        
        try:
            if self._openai_client is not None:
                # New client (>=1.0)
                # Some newer models (e.g., gpt-5.*) do not accept 'max_tokens' and require 'max_completion_tokens'.
                # We keep backward compatibility by allowing users to configure max_tokens and mapping when needed.
                requested_max_tokens = kwargs.get("max_tokens", self.max_tokens)
                requested_max_completion_tokens = kwargs.get("max_completion_tokens", self.max_completion_tokens)

                def _chat_completion(local_messages):
                    create_kwargs: Dict[str, Any] = {
                        "model": self.model_name,
                        "messages": local_messages,
                        "temperature": kwargs.get("temperature", self.temperature),
                    }

                    # Heuristic mapping rule:
                    # - If caller explicitly provides max_completion_tokens -> use it
                    # - Else if model name looks like gpt-5.* -> use max_completion_tokens (fall back to max_tokens value)
                    # - Else -> use legacy max_tokens
                    if requested_max_completion_tokens is not None:
                        create_kwargs["max_completion_tokens"] = requested_max_completion_tokens
                    elif str(self.model_name).startswith("gpt-5"):
                        create_kwargs["max_completion_tokens"] = requested_max_tokens
                    else:
                        create_kwargs["max_tokens"] = requested_max_tokens
                    return self._openai_client.chat.completions.create(**create_kwargs)

                try:
                    response = _chat_completion(messages)
                except Exception as first_error:
                    # vLLM/OpenAI-compatible fallback:
                    # Some chat templates reject system messages and require strict
                    # user/assistant alternation. If that happens, retry once by
                    # flattening system prompt into a single user turn.
                    needs_alt_roles = "Conversation roles must alternate" in str(first_error)
                    can_retry_flatten = bool(
                        self.base_url and system_prompt and not used_flattened_system and needs_alt_roles
                    )
                    if not can_retry_flatten:
                        raise
                    logger.warning(
                        "OpenAIModel: retrying with flattened system prompt due to role alternation constraint from base_url backend."
                    )
                    retry_messages = [{
                        "role": "user",
                        "content": (
                            f"[System Instruction]\n{system_prompt}\n\n"
                            f"[User Question]\n{prompt}"
                        ),
                    }]
                    response = _chat_completion(retry_messages)
                return response.choices[0].message.content

            raise RuntimeError(
                "OpenAI client is not initialized. Ensure openai>=1.0.0 is installed and an API key is available."
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI client is available."""
        if self._openai is None:
            logger.debug("OpenAIModel.is_available: False - openai package not imported")
            return False
        if not self.api_key and not self.allow_empty_api_key:
            logger.debug("OpenAIModel.is_available: False - API key not found")
            return False
        logger.debug(f"OpenAIModel.is_available: True - model={self.model_name}")
        return True

