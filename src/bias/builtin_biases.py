"""
Built-in bias implementations.
"""
import re
from typing import Dict, Any, Optional

from .base import BaseBias


class JargonOverloadingBias(BaseBias):
    """Inject excessive medical jargon into text."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("jargon_overloading", config)
        # Medical jargon terms to inject
        self.jargon_terms = [
            "pathophysiology", "etiology", "pathogenesis", "pathology",
            "differential diagnosis", "clinical presentation", "diagnostic criteria",
            "therapeutic intervention", "pharmacological management",
            "prognostic indicators", "epidemiological factors", "pathophysiological mechanisms"
        ]
    
    def apply(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject medical jargon into text.
        
        Args:
            text: Original text
            context: Optional context
            
        Returns:
            Text with jargon injected
        """
        # Simple rule-based injection: add jargon terms at strategic points
        modified = text
        
        # Add jargon at the beginning
        if not any(term in modified.lower() for term in self.jargon_terms):
            jargon_intro = "From a pathophysiological perspective, "
            if not modified.startswith(jargon_intro):
                modified = jargon_intro + modified
        
        # Replace simple terms with jargon
        replacements = {
            "cause": "etiological factor",
            "reason": "pathophysiological mechanism",
            "treatment": "therapeutic intervention",
            "medicine": "pharmacological agent",
            "disease": "pathological condition",
            "symptom": "clinical manifestation",
            "test": "diagnostic procedure",
            "result": "diagnostic finding"
        }
        
        for simple, jargon in replacements.items():
            # Replace whole words only
            pattern = r'\b' + re.escape(simple) + r'\b'
            modified = re.sub(pattern, jargon, modified, flags=re.IGNORECASE)
        
        # Add a jargon-heavy conclusion if not present
        if "pathophysiological" not in modified.lower()[-100:]:
            modified += " This reflects the underlying pathophysiological mechanisms and requires comprehensive diagnostic evaluation."
        
        return modified

    def apply_word_replacement(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Only replace words/phrases with medical jargon without restructuring.
        """
        modified = text
        replacements = {
            "cause": "etiological factor",
            "reason": "pathophysiological mechanism",
            "treatment": "therapeutic intervention",
            "medicine": "pharmacological agent",
            "disease": "pathological condition",
            "symptom": "clinical manifestation",
            "test": "diagnostic procedure",
            "result": "diagnostic finding"
        }
        for simple, jargon in replacements.items():
            pattern = r'\b' + re.escape(simple) + r'\b'
            modified = re.sub(pattern, jargon, modified, flags=re.IGNORECASE)
        return modified


class ComplexityBias(BaseBias):
    """Make text unnecessarily complex."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("complexity", config)
    
    def apply(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Add unnecessary complexity."""
        # Add complex sentence structures
        modified = "It is important to note that " + text.lower()
        modified = modified.replace(". ", ". Furthermore, it should be emphasized that ")
        return modified


class AuthorityBias(BaseBias):
    """Add excessive authority references."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("authority", config)
    
    def apply(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Add authority references."""
        modified = "According to established medical literature and clinical guidelines, " + text
        modified += " This is supported by numerous peer-reviewed studies and expert consensus."
        return modified


# Registry of built-in biases
BUILTIN_BIASES = {
    "jargon_overloading": JargonOverloadingBias,
    "complexity": ComplexityBias,
    "authority": AuthorityBias,
}


