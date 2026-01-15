"""
Base classes for model interfaces.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """Base class for all model interfaces."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.
        
        Args:
            model_name: Name of the model
            config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if model is available for use.
        
        Returns:
            True if model can be used
        """
        pass

