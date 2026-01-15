"""
Prompt templates for judge models.
"""
from typing import Dict, Any, Optional


def build_pairwise_judge_prompt(
    question: str,
    answer_A: str,
    answer_B: str,
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None
) -> tuple[str, str]:
    """
    Build prompt for pairwise judgment.
    
    Args:
        question: The question
        answer_A: First answer
        answer_B: Second answer
        system_prompt: Optional system prompt
        user_template: Optional user prompt template
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    default_system = (
        "You are an expert medical evaluator. Compare two answers to a medical question "
        "and determine which one is better. Consider accuracy, clarity, completeness, "
        "and appropriateness for the target audience."
    )
    
    default_user = (
        f"Question: {question}\n\n"
        f"Answer A: {answer_A}\n\n"
        f"Answer B: {answer_B}\n\n"
        "Which answer is better? Provide your judgment with reasoning."
    )
    
    system = system_prompt or default_system
    user = user_template or default_user
    
    if user_template:
        user = user_template.format(
            question=question,
            answer_a=answer_A,
            answer_b=answer_B
        )
    
    return system, user


