"""
Judge implementations.
"""
import re
import json
from typing import Dict, Any, Optional
from loguru import logger

from .base import BaseJudge, JudgeResult
from .prompts import build_pairwise_judge_prompt
from ..models.base import BaseModel
from ..models.registry import ModelRegistry


class MockJudge(BaseJudge):
    """Mock judge using heuristic rules."""
    
    def __init__(self, model_name: str = "mock-judge", config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
    
    def judge_pairwise(self, question: str, answer_A: str, answer_B: str) -> JudgeResult:
        """
        Judge using simple heuristics.
        
        Heuristics:
        - Longer answer is better (if difference > 50 chars)
        - Answer with "因此" or "建议" is better
        - Answer with more medical terms is better
        - Otherwise, prefer A
        """
        # Heuristic 1: Length
        len_A = len(answer_A)
        len_B = len(answer_B)
        
        # Heuristic 2: Keywords
        keywords = ["因此", "建议", "应该", "需要", "重要"]
        score_A = sum(1 for kw in keywords if kw in answer_A)
        score_B = sum(1 for kw in keywords if kw in answer_B)
        
        # Heuristic 3: Medical terms
        medical_terms = ["治疗", "诊断", "症状", "疾病", "药物", "检查"]
        med_A = sum(1 for term in medical_terms if term in answer_A)
        med_B = sum(1 for term in medical_terms if term in answer_B)
        
        # Calculate total score
        total_A = len_A * 0.1 + score_A * 2 + med_A * 1.5
        total_B = len_B * 0.1 + score_B * 2 + med_B * 1.5
        
        # Determine winner
        if abs(total_A - total_B) < 0.5:
            winner = "tie"
        elif total_A > total_B:
            winner = "A"
        else:
            winner = "B"
        
        explanation = (
            f"Mock judgment: Answer {winner} selected based on length, keywords, and medical terms. "
            f"Scores: A={total_A:.2f}, B={total_B:.2f}"
        )
        
        return JudgeResult(
            winner=winner,
            score_A=total_A,
            score_B=total_B,
            explanation=explanation,
            raw={"heuristic_scores": {"A": total_A, "B": total_B}}
        )
    
    def is_available(self) -> bool:
        """Mock judge is always available."""
        return True


class ModelJudge(BaseJudge):
    """Judge using a language model."""
    
    def __init__(
        self,
        model_type: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        """
        Initialize model-based judge.
        
        Args:
            model_type: Type of model ("openai", "vllm")
            model_name: Name of the model
            config: Model configuration
            system_prompt: Optional system prompt for pairwise judging
            user_template: Optional user prompt template for pairwise judging
        """
        super().__init__(model_name, config)
        self.model_type = model_type
        self.model_config = config or {}
        self.system_prompt = system_prompt
        self.user_template = user_template
        
        # Initialize model
        allow_fallback = self.model_config.get("allow_fallback_mock", False)
        
        try:
            logger.info(f"ModelJudge: Initializing {model_type} model '{model_name}'")
            self.model = ModelRegistry.create_model(
                model_type=model_type,
                model_name=model_name,
                config=self.model_config
            )
            
            if not self.model.is_available():
                error_msg = f"Judge model {model_type}:{model_name} is not available"
                if allow_fallback:
                    logger.warning(f"{error_msg}, falling back to mock judge (allow_fallback_mock=True)")
                    self.model = None
                else:
                    raise RuntimeError(
                        f"{error_msg}. Set 'allow_fallback_mock: true' in judge config to allow fallback."
                    )
            else:
                logger.info(f"ModelJudge: Successfully initialized {model_type} model '{model_name}'")
                
        except RuntimeError:
            # Re-raise RuntimeError (from ModelRegistry) - these are explicit errors
            raise
        except Exception as e:
            error_msg = f"Error initializing judge model {model_type}:{model_name}: {e}"
            logger.error(error_msg)
            if allow_fallback:
                logger.warning("Falling back to mock judge (allow_fallback_mock=True)")
                self.model = None
            else:
                raise RuntimeError(
                    f"{error_msg}. Set 'allow_fallback_mock: true' in judge config to allow fallback."
                ) from e
    
    def judge_pairwise(self, question: str, answer_A: str, answer_B: str) -> JudgeResult:
        """
        Judge using language model.
        
        Args:
            question: The question
            answer_A: First answer
            answer_B: Second answer
            
        Returns:
            JudgeResult
        """
        if self.model is None:
            # This should not happen if allow_fallback_mock is False (would have raised error in __init__)
            # But if it does happen, we still need to handle it
            logger.error("ModelJudge.judge_pairwise: self.model is None - this should not happen!")
            logger.warning("Falling back to mock judge for this judgment")
            mock_judge = MockJudge()
            return mock_judge.judge_pairwise(question, answer_A, answer_B)
        
        # Build prompt
        system_prompt, user_prompt = build_pairwise_judge_prompt(
            question=question,
            answer_A=answer_A,
            answer_B=answer_B,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
        )
        
        try:
            response = self.model.generate(user_prompt, system_prompt=system_prompt)
            
            # Check if response indicates quota error or other blocking errors
            if isinstance(response, str):
                response_lower = response.lower().strip()
                # If the response already looks like a valid judgment output,
                # do not treat generic words (e.g., "blocked conduction") as runtime errors.
                looks_like_valid_judgment = any(
                    marker in response_lower
                    for marker in ("verdict:", "[[a]]", "[[b]]", '"winner"')
                )

                quota_error_patterns = [
                    "quota exceeded",
                    "rate limit exceeded",
                    "insufficient_quota",
                ]
                timeout_error_patterns = [
                    "request timed out",
                    "timed out",
                    "timeout after",
                    "context deadline exceeded",
                    "deadline exceeded",
                    "read timeout",
                    "connection timed out",
                ]
                blocked_error_patterns = [
                    "blocked by safety",
                    "blocked by policy",
                    "content blocked",
                    "safety filter",
                    "policy violation",
                    "violates safety policy",
                ]

                if any(p in response_lower for p in quota_error_patterns):
                    logger.error(f"Quota error detected in response: {response}")
                    allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
                    if allow_fallback:
                        logger.warning("Using mock judge for this judgment due to quota error (allow_fallback_mock=True)")
                        mock_judge = MockJudge()
                        return mock_judge.judge_pairwise(question, answer_A, answer_B)
                    else:
                        # Return invalid result for quota errors to prevent metric pollution
                        logger.warning("Quota error but allow_fallback_mock=False. Returning invalid result (will be excluded from metrics).")
                        return JudgeResult(
                            winner="tie",
                            score_A=0.0,
                            score_B=0.0,
                            explanation=f"Quota error: {response}",
                            raw={"model_response": response, "error": "quota_exceeded"},
                            is_valid=False  # Mark as invalid to exclude from metrics
                        )
                elif (
                    not looks_like_valid_judgment
                    and (
                        any(p in response_lower for p in timeout_error_patterns)
                        or any(p in response_lower for p in blocked_error_patterns)
                    )
                ):
                    logger.warning(f"Error response detected: {response}")
                    allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
                    if allow_fallback:
                        logger.warning("Using mock judge for this judgment due to error response (allow_fallback_mock=True)")
                        mock_judge = MockJudge()
                        return mock_judge.judge_pairwise(question, answer_A, answer_B)
                    else:
                        # Return invalid result for timeout/blocked errors to prevent metric pollution
                        # This will be excluded from final metrics calculation
                        error_type = "safety_filter" if any(p in response_lower for p in blocked_error_patterns) else "timeout_or_blocked"
                        logger.warning(f"Error response but allow_fallback_mock=False. Returning invalid result (will be excluded from metrics). Error: {error_type}")
                        return JudgeResult(
                            winner="tie",
                            score_A=0.0,
                            score_B=0.0,
                            explanation=f"Error: {response}",
                            raw={"model_response": response, "error": error_type},
                            is_valid=False  # Mark as invalid to exclude from metrics
                        )
            
            # Parse response to extract winner, scores, and explanation
            winner, score_A, score_B, explanation = self._parse_response(response, answer_A, answer_B)
            
            return JudgeResult(
                winner=winner,
                score_A=score_A,
                score_B=score_B,
                explanation=explanation,
                raw={"model_response": response}
            )
        except Exception as e:
            logger.error(f"Error in model judgment: {e}")
            # Check if it's a quota error
            error_str = str(e)
            is_quota_error = "429" in error_str or ("ResourceExhausted" in type(e).__name__ and "quota" in error_str.lower())
            
            allow_fallback = bool(self.model_config.get("allow_fallback_mock", False))
            if allow_fallback:
                logger.warning("Using mock judge for this judgment due to runtime error (allow_fallback_mock=True)")
                mock_judge = MockJudge()
                return mock_judge.judge_pairwise(question, answer_A, answer_B)
            elif is_quota_error:
                # For quota errors, return invalid tie result to allow continuation
                # but exclude from metrics (is_valid=False).
                logger.warning("Quota error but allow_fallback_mock=False. Returning invalid result to allow continuation.")
                return JudgeResult(
                    winner="tie",
                    score_A=0.0,
                    score_B=0.0,
                    explanation=f"Quota error: {error_str}",
                    raw={"error": "quota_exceeded", "exception": str(e)},
                    is_valid=False,
                )
            raise
    
    def _parse_response(self, response: str, answer_A: str, answer_B: str) -> tuple[str, Optional[float], Optional[float], str]:
        """
        Parse model response to extract judgment.
        Supports both JSON format and free-text format for backward compatibility.
        
        Args:
            response: Model response text
            answer_A: First answer (for context)
            answer_B: Second answer (for context)
            
        Returns:
            Tuple of (winner, score_A, score_B, explanation)
        """
        # Try to parse as JSON first (new format)
        try:
            # Extract JSON from response (may be wrapped in markdown code blocks)
            json_text = response.strip()
            # Remove markdown code blocks if present
            if json_text.startswith("```"):
                # Extract content between ```json and ```
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1).strip()
                else:
                    # Try to extract JSON from between first { and last }
                    json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
            
            # Parse JSON
            parsed = json.loads(json_text)
            
            winner = parsed.get("winner", "tie").upper()
            if winner not in ["A", "B", "TIE"]:
                winner = "tie"
            elif winner == "TIE":
                winner = "tie"
            
            explanation = parsed.get("explanation", "")
            
            # Scores are not in new format, set to None
            score_A = None
            score_B = None
            
            logger.debug(f"Successfully parsed JSON response: winner={winner}")
            return winner, score_A, score_B, explanation
            
        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            # Fall back to text-based parsing (supports both new format and old format)
            logger.debug(f"Failed to parse as JSON, falling back to text parsing: {e}")
            response_lower = response.lower()
            
            # Try structured format: "Verdict: [[A]]" / "[[B]]" and "Explanation:"
            winner = "tie"
            explanation = ""
            
            # Extract verdict from "[[A]]" or "[[B]]" format (order-independent)
            verdict_match = re.search(r'\[\[([AB])\]\]', response, re.IGNORECASE)
            
            # Extract explanation: capture everything after "Explanation:" until
            # "Verdict:" (old order) or end-of-string (new order: verdict first)
            explanation_match = re.search(
                r'explanation:\s*(.+?)(?=\nverdict:|\Z)',
                response, re.IGNORECASE | re.DOTALL
            )
            if explanation_match:
                explanation = explanation_match.group(1).strip()
            if verdict_match:
                winner = verdict_match.group(1).upper()
            else:
                # Fall back to old format parsing for backward compatibility
                if "answer a" in response_lower or "answer_a" in response_lower or "a is better" in response_lower:
                    winner = "A"
                elif "answer b" in response_lower or "answer_b" in response_lower or "b is better" in response_lower:
                    winner = "B"
                elif "tie" in response_lower or "equal" in response_lower or "both" in response_lower:
                    winner = "tie"
                
                # If no explanation extracted, use first 500 chars as fallback
                if not explanation:
                    explanation = response[:500].strip()
            
            # Try to extract scores (for old format compatibility)
            score_A = None
            score_B = None
            
            score_pattern = r'score[_\s]*(?:a|1)[:\s]*(\d+\.?\d*)'
            match = re.search(score_pattern, response_lower)
            if match:
                score_A = float(match.group(1))
            
            score_pattern = r'score[_\s]*(?:b|2)[:\s]*(\d+\.?\d*)'
            match = re.search(score_pattern, response_lower)
            if match:
                score_B = float(match.group(1))
            
            return winner, score_A, score_B, explanation
    
    def is_available(self) -> bool:
        """Check if judge model is available."""
        return self.model is not None and self.model.is_available()


def create_judge(
    judge_type: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> BaseJudge:
    """
    Create a judge instance.
    
    Args:
        judge_type: Type of judge ("mock", "openai", "vllm", "gemini", "anthropic")
        model_name: Name of the model
        config: Judge configuration
        
    Returns:
        Judge instance
    """
    if judge_type == "mock":
        return MockJudge(model_name=model_name, config=config)
    elif judge_type in ["openai", "vllm", "gemini", "anthropic", "hf"]:
        return ModelJudge(
            model_type=judge_type,
            model_name=model_name,
            config=config,
            system_prompt=system_prompt,
            user_template=user_template,
        )
    else:
        raise ValueError(f"Unknown judge type: {judge_type}. Available: mock, openai, vllm, gemini, anthropic")

