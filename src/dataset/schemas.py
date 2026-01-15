"""
Data schemas for medical bias evaluation dataset.
"""
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PairwiseSample:
    """Schema for pairwise comparison samples."""
    id: str
    question: str
    answer_1: str
    answer_2: str
    preferred: Optional[str] = None  # "1", "2", or None
    meta: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate preferred field."""
        if self.preferred is not None and self.preferred not in ["1", "2"]:
            raise ValueError(f"preferred must be '1', '2', or None, got {self.preferred}")
        
        if self.meta is None:
            self.meta = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer_1": self.answer_1,
            "answer_2": self.answer_2,
            "preferred": self.preferred,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PairwiseSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            answer_1=data["answer_1"],
            answer_2=data["answer_2"],
            preferred=data.get("preferred"),
            meta=data.get("meta", {})
        )
    
    def get_gt_answer(self) -> tuple[str, str]:
        """
        Get the ground truth answer.
        Returns: (answer_text, answer_key) where answer_key is "1" or "2"
        """
        if self.preferred == "1":
            return self.answer_1, "1"
        elif self.preferred == "2":
            return self.answer_2, "2"
        else:
            # Placeholder: assume answer_1 is GT
            return self.answer_1, "1"
    
    def get_non_gt_answer(self) -> tuple[str, str]:
        """
        Get the non-ground truth answer.
        Returns: (answer_text, answer_key) where answer_key is "1" or "2"
        """
        if self.preferred == "1":
            return self.answer_2, "2"
        elif self.preferred == "2":
            return self.answer_1, "1"
        else:
            # Placeholder: assume answer_2 is non-GT
            return self.answer_2, "2"


@dataclass
class ScalarSample:
    """Schema for scalar evaluation samples (future use)."""
    id: str
    question: str
    answer: str
    score: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.meta is None:
            self.meta = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "score": self.score,
            "meta": self.meta
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalarSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            score=data.get("score"),
            meta=data.get("meta", {})
        )

