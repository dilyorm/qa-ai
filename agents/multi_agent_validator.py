"""Multi-Agent Validator for orchestrating answerer-validator consensus loop."""

import logging
from typing import Optional
from agents.answerer_agent import AnswererAgent
from agents.validator_agent import ValidatorAgent
from models.schemas import QuestionItem


logger = logging.getLogger(__name__)


class ValidationResult:
    """Result from the multi-agent validation process."""
    
    def __init__(self, selected_answer: str, iterations: int, consensus_reached: bool):
        """
        Initialize a validation result.
        
        Args:
            selected_answer: The final selected answer
            iterations: Number of validation iterations performed
            consensus_reached: Whether agents reached consensus
        """
        self.selected_answer = selected_answer
        self.iterations = iterations
        self.consensus_reached = consensus_reached


class MultiAgentValidator:
    """Orchestrates the answerer-validator consensus loop."""
    
    def __init__(
        self,
        answerer_agent: AnswererAgent,
        validator_agent: ValidatorAgent,
        max_iterations: int = 5
    ):
        """
        Initialize the Multi-Agent Validator.
        
        Args:
            answerer_agent: The answerer agent instance
            validator_agent: The validator agent instance
            max_iterations: Maximum number of validation iterations (default: 5)
        """
        self.answerer_agent = answerer_agent
        self.validator_agent = validator_agent
        self.max_iterations = max_iterations
        logger.info(f"MultiAgentValidator initialized with max_iterations={max_iterations}")
    
    def validate_question(self, question: QuestionItem) -> ValidationResult:
        """
        Validate a question through the multi-agent consensus loop.
        
        The validation loop:
        1. Answerer selects an answer
        2. Validator reviews the selection
        3. If validator agrees, return the answer (consensus reached)
        4. If validator disagrees:
           - If max iterations reached, return most recent answer with warning
           - Otherwise, pass criticism to answerer and repeat from step 1
        
        Args:
            question: The question to validate
            
        Returns:
            ValidationResult with selected answer, iteration count, and consensus status
            
        Raises:
            Exception: If agent calls fail
        """
        logger.info(
            f"Starting validation for question {question.questionNumber}",
            extra={'question_number': question.questionNumber}
        )
        
        iteration = 0
        previous_answer: Optional[str] = None
        criticism: Optional[str] = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(
                f"Question {question.questionNumber}: Iteration {iteration}/{self.max_iterations}",
                extra={'question_number': question.questionNumber}
            )
            
            # Step 1: Get answer from answerer agent
            answerer_response = self.answerer_agent.answer_question(
                question,
                previous_answer=previous_answer,
                criticism=criticism
            )
            
            current_answer = answerer_response.selected_answer
            reasoning = answerer_response.reasoning
            
            # Step 2: Get validation from validator agent
            validator_response = self.validator_agent.validate_answer(
                question,
                current_answer,
                reasoning
            )
            
            # Step 3: Check for consensus
            if validator_response.agrees():
                logger.info(
                    f"Question {question.questionNumber}: Consensus reached after {iteration} iteration(s)",
                    extra={'question_number': question.questionNumber}
                )
                return ValidationResult(
                    selected_answer=current_answer,
                    iterations=iteration,
                    consensus_reached=True
                )
            
            # Step 4: Handle disagreement
            logger.debug(
                f"Question {question.questionNumber}: Validator disagreed - {validator_response.criticism}",
                extra={'question_number': question.questionNumber}
            )
            
            # Prepare for next iteration
            previous_answer = current_answer
            criticism = validator_response.criticism
        
        # Max iterations reached without consensus
        logger.warning(
            f"Question {question.questionNumber}: Max iterations ({self.max_iterations}) "
            f"reached without consensus. Accepting most recent answer: {current_answer}",
            extra={'question_number': question.questionNumber}
        )
        
        return ValidationResult(
            selected_answer=current_answer,
            iterations=iteration,
            consensus_reached=False
        )
