"""
Base classes for judge models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Literal


class JudgeResult:
    """Result from judge evaluation."""
    
    def __init__(
        self,
        winner: Literal["A", "B", "tie"],
        score_A: Optional[float] = None,
        score_B: Optional[float] = None,
        explanation: Optional[str] = None,
        raw: Optional[Dict[str, Any]] = None,
        is_valid: bool = True
    ):
        """
        Initialize judge result.
        
        Args:
            winner: "A", "B", or "tie"
            score_A: Optional score for answer A
            score_B: Optional score for answer B
            explanation: Optional explanation
            raw: Raw response from judge
            is_valid: Whether this judgment is valid (False for safety filter blocks, quota errors, etc.)
        """
        self.winner = winner
        self.score_A = score_A
        self.score_B = score_B
        self.explanation = explanation
        self.raw = raw or {}
        self.is_valid = is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "winner": self.winner,
            "score_A": self.score_A,
            "score_B": self.score_B,
            "explanation": self.explanation,
            "raw": self.raw,
            "is_valid": self.is_valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeResult":
        """Create from dictionary."""
        return cls(
            winner=data["winner"],
            score_A=data.get("score_A"),
            score_B=data.get("score_B"),
            explanation=data.get("explanation"),
            raw=data.get("raw", {}),
            is_valid=data.get("is_valid", True)
        )


class BaseJudge(ABC):
    """Base class for judge models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize judge.
        
        Args:
            model_name: Name of the judge model
            config: Judge configuration
        """
        self.model_name = model_name
        self.config = config or {}
    
    @abstractmethod
    def judge_pairwise(
        self,
        question: str,
        answer_A: str,
        answer_B: str
    ) -> JudgeResult:
        """
        Judge which answer is better.
        
        Args:
            question: The question
            answer_A: First answer
            answer_B: Second answer
            
        Returns:
            JudgeResult with winner and scores
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if judge is available."""
        pass


