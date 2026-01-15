"""
Base classes for bias injection.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseBias(ABC):
    """Base class for bias types."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize bias.
        
        Args:
            name: Name of the bias type
            config: Bias configuration
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def apply(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Apply bias to text.
        
        Args:
            text: Text to inject bias into
            context: Optional context (e.g., question)
            
        Returns:
            Text with bias injected
        """
        pass
    
    def get_description(self) -> str:
        """Get description of the bias."""
        return self.config.get("description", f"Bias type: {self.name}")


