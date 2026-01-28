"""
Model registry for creating model instances.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from loguru import logger

from .base import BaseModel
from .mock_client import MockModel
from .openai_client import OpenAIModel
from .anthropic_client import AnthropicModel
from .gemini_client import GeminiModel
from .hf_client import HuggingFaceModel


class ModelRegistry:
    """Registry for model creation."""
    
    _models = {
        "mock": MockModel,
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "gemini": GeminiModel,
        "hf": HuggingFaceModel,
    }

    # A "model pool" loaded from configs/models.yaml.
    # Expected (new) shape: models_config[provider][model_id] -> config dict, with optional provider-level "defaults".
    # Backward compatible (old) shape: models_config[provider] -> flat config dict.
    _models_config: Dict[str, Any] = {}

    _RESERVED_POOL_KEYS = {"defaults"}
    _MODEL_REF_KEYS = {"provider", "type", "model_id", "model_name", "model"}

    @classmethod
    def set_models_config(cls, models_config: Optional[Dict[str, Any]]) -> None:
        """Set the model pool config (usually loaded from configs/models.yaml)."""
        cls._models_config = models_config or {}
    
    @classmethod
    def infer_provider_from_model_id(cls, model_id: str) -> Optional[str]:
        """
        Infer provider from model_id by searching through the model pool.
        
        Args:
            model_id: Model identifier (e.g., "gpt4omini", "gemini3_pro")
            
        Returns:
            Provider name if found, None otherwise
        """
        if not model_id or not cls._models_config:
            return None
        
        # Search through all providers
        for provider, provider_cfg in cls._models_config.items():
            if not isinstance(provider_cfg, dict):
                continue
            
            # Check if model_id exists in this provider (excluding reserved keys)
            non_reserved_items = {k: v for k, v in provider_cfg.items() if k not in cls._RESERVED_POOL_KEYS}
            is_pool = any(isinstance(v, dict) for v in non_reserved_items.values())
            
            if is_pool:
                # Model pool format: check if model_id exists
                if model_id in provider_cfg and model_id not in cls._RESERVED_POOL_KEYS:
                    return provider
            else:
                # Flat format: model_id might be the actual model name, but we can't reliably infer
                # Skip for now - user should specify provider explicitly
                pass
        
        return None

    @classmethod
    def resolve_model_config(
        cls,
        provider: str,
        model_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Resolve (model_name, config) by looking up provider+model_id from the model pool.

        - New models.yaml: models[provider][model_id] with optional models[provider]["defaults"]
        - Old models.yaml: models[provider] is a flat config dict (we treat model_id as the actual model name)

        Args:
            provider: e.g. "openai", "hf", "mock"
            model_id: a stable identifier in experiment.yaml (new) or an actual model name (old)
            overrides: optional per-run overrides (e.g. allow_fallback_mock)

        Returns:
            (resolved_model_name, merged_config)
        """
        overrides = (overrides or {}).copy()
        # Drop selection keys from overrides to avoid confusing merges.
        for k in list(overrides.keys()):
            if k in cls._MODEL_REF_KEYS:
                overrides.pop(k, None)

        provider_cfg = cls._models_config.get(provider)
        merged: Dict[str, Any] = {}

        if isinstance(provider_cfg, dict) and provider_cfg:
            # Detect "model pool" if any value (excluding defaults) is a dict (i.e., model_id -> dict config).
            non_reserved_items = {k: v for k, v in provider_cfg.items() if k not in cls._RESERVED_POOL_KEYS}
            is_pool = any(isinstance(v, dict) for v in non_reserved_items.values())

            if is_pool:
                if model_id not in provider_cfg or model_id in cls._RESERVED_POOL_KEYS:
                    available = [k for k in provider_cfg.keys() if k not in cls._RESERVED_POOL_KEYS]
                    raise KeyError(
                        f"Unknown model_id '{model_id}' for provider '{provider}'. "
                        f"Available model_id: {available}"
                    )
                defaults = provider_cfg.get("defaults") or {}
                if defaults and not isinstance(defaults, dict):
                    raise TypeError(f"Invalid models.yaml: {provider}.defaults must be a dict")
                model_cfg = provider_cfg.get(model_id) or {}
                if model_cfg and not isinstance(model_cfg, dict):
                    raise TypeError(f"Invalid models.yaml: {provider}.{model_id} must be a dict")
                merged = {**defaults, **model_cfg}
            else:
                # Flat provider config (old style): treat model_id as the actual model name.
                merged = provider_cfg.copy()
        else:
            merged = {}

        # Apply explicit overrides last.
        merged.update(overrides)

        # Determine the actual model name to pass into client constructors.
        resolved_model_name = (
            merged.get("model_name")
            or merged.get("model")
            or model_id
        )

        # Normalize common aliases to keep downstream code stable.
        # - OpenAI client uses the positional model_name; we also keep merged["model"] for directory naming/logging.
        if provider == "openai":
            merged.setdefault("model", resolved_model_name)
            merged.setdefault("model_name", resolved_model_name)

        return resolved_model_name, merged
    
    @classmethod
    def create_model(cls, model_type: str, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ("mock", "openai", "hf")
            model_name: Name of the model (backward compatible) OR model_id (when using model pool)
            config: Model configuration overrides (optional). If a model pool is set, the final config is:
                models.yaml[model_type][model_name] (or flat models.yaml[model_type]) merged with overrides.
            
        Returns:
            Model instance
        """
        # Resolve from model pool if available; otherwise treat model_name as the actual model name.
        try:
            resolved_model_name, resolved_config = cls.resolve_model_config(
                provider=model_type,
                model_id=model_name,
                overrides=config,
            )
        except KeyError:
            # If there's a model pool but the model_id is unknown, surface the error (don't silently fall back).
            raise
        except Exception as e:
            # If resolution fails for any other reason, fall back to old behavior.
            logger.warning(f"Model config resolution failed, falling back to direct config usage: {e}")
            resolved_model_name, resolved_config = model_name, (config or {})

        # Log model creation attempt
        logger.info("ModelRegistry.create_model called:")
        logger.info(f"  model_type: {model_type}")
        logger.info(f"  model_id/name: {model_name}")
        logger.info(f"  resolved_model_name: {resolved_model_name}")
        logger.info(f"  config keys: {list(resolved_config.keys()) if resolved_config else []}")
        
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        allow_fallback = resolved_config.get("allow_fallback_mock", False) if resolved_config else False
        
        try:
            model = model_class(model_name=resolved_model_name, config=resolved_config)
            logger.info(f"Created {model_type} model instance: {resolved_model_name}")
            
            if not model.is_available():
                error_msg = (
                    f"Model {model_type}:{resolved_model_name} is not available. "
                    f"Check API key, package installation, and configuration."
                )
                
                if allow_fallback:
                    logger.warning(f"{error_msg} Falling back to mock (allow_fallback_mock=True).")
                    return MockModel(model_name="fallback-mock", config={"deterministic": True})
                else:
                    logger.error(error_msg)
                    raise RuntimeError(
                        f"Model {model_type}:{resolved_model_name} is not available. "
                        f"Set 'allow_fallback_mock: true' in config to allow fallback to mock."
                    )
            
            logger.info(f"Model {model_type}:{resolved_model_name} is available and ready to use")
            return model
        except Exception as e:
            error_msg = f"Error creating model {model_type}:{resolved_model_name}: {e}"
            logger.error(error_msg)
            
            if allow_fallback:
                logger.warning("Falling back to mock model (allow_fallback_mock=True)")
                return MockModel(model_name="fallback-mock", config={"deterministic": True})
            else:
                raise RuntimeError(
                    f"{error_msg}. Set 'allow_fallback_mock: true' in config to allow fallback to mock."
                ) from e

