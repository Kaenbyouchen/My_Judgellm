"""
Mock model client for testing without external dependencies.
"""
import random
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseModel


class MockModel(BaseModel):
    """Mock model that returns deterministic or random responses."""
    
    def __init__(self, model_name: str = "mock-model", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.deterministic = config.get("deterministic", True) if config else True
        self._counter = 0
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate mock response.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Mock generated text
        """
        self._counter += 1
        
        if self.deterministic:
            # Deterministic mock: return based on prompt hash
            seed = hash(prompt) % 1000
            responses = [
                "This is a mock response with some medical terminology.",
                "Based on the provided information, the answer includes technical details.",
                "The response contains relevant medical information and clinical considerations.",
                "Here is a comprehensive answer with appropriate medical context.",
                "This answer provides detailed medical information with proper terminology."
            ]
            idx = seed % len(responses)
            base_response = responses[idx]
            
            # Add some variation based on prompt length
            if len(prompt) > 200:
                base_response += " Additional details are provided based on the complexity of the question."
            
            return base_response
        else:
            # Random mock
            templates = [
                "Mock response: {prompt[:50]}...",
                "Generated answer with medical context.",
                "Response includes relevant information."
            ]
            return random.choice(templates).format(prompt=prompt)
    
    def is_available(self) -> bool:
        """Mock model is always available."""
        return True

