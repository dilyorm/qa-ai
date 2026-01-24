"""Validator Agent for critically reviewing answer selections."""

import logging
from typing import Optional
from agents.gemini_client import GeminiClient
from models.schemas import QuestionItem


logger = logging.getLogger(__name__)


class ValidatorResponse:
    """Response from the Validator Agent."""
    
    def __init__(self, verdict: str, criticism: Optional[str] = None):
        """
        Initialize a validator response.
        
        Args:
            verdict: Either "AGREE" or "DISAGREE"
            criticism: Criticism of the answer (present when verdict is DISAGREE)
        """
        if verdict not in ["AGREE", "DISAGREE"]:
            raise ValueError(f"Invalid verdict: {verdict}. Must be AGREE or DISAGREE")
        
        self.verdict = verdict
        self.criticism = criticism
    
    def agrees(self) -> bool:
        """Check if the validator agrees with the answer."""
        return self.verdict == "AGREE"


class ValidatorAgent:
    """Agent responsible for critically reviewing and validating answer selections."""
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the Validator Agent.
        
        Args:
            gemini_client: The Gemini API client to use for generating responses
        """
        self.gemini_client = gemini_client
        logger.info("ValidatorAgent initialized")
    
    def _build_prompt(
        self,
        question: QuestionItem,
        selected_answer: str,
        reasoning: str
    ) -> str:
        """
        Build the prompt for the validator agent.
        
        Args:
            question: The question being answered
            selected_answer: The answer selected by the answerer (letter)
            reasoning: The answerer's reasoning
            
        Returns:
            The formatted prompt string
        """
        # Format answer options as lettered list (A, B, C, D, etc.)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        answer_list = "\n".join([f"{letters[i]}. {answer.content}" for i, answer in enumerate(question.answers)])
        
        prompt = f"""You are a critical reviewer evaluating answer selections for multiple-choice questions.

Question Content:
{question.content}

Question Title:
{question.title}

Available Answers:
{answer_list}

Proposed Answer:
{selected_answer}

Answerer's Reasoning:
{reasoning}

Task: Critically evaluate whether the proposed answer is the best choice. Consider:
- Does it accurately address the question?
- Are there better options available?
- Is the reasoning sound?

Format your response as:
VERDICT: [AGREE or DISAGREE]
"""
        
        # Add instruction for criticism if disagreeing
        prompt += """If you DISAGREE, also provide:
CRITICISM: [specific issues and suggested alternative letter]
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> ValidatorResponse:
        """
        Parse the agent's response to extract verdict and criticism.
        
        Args:
            response_text: The raw response from the Gemini API
            
        Returns:
            ValidatorResponse with verdict and optional criticism
            
        Raises:
            ValueError: If response cannot be parsed
        """
        lines = response_text.strip().split('\n')
        verdict = None
        criticism = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("VERDICT:"):
                verdict_text = line[len("VERDICT:"):].strip().upper()
                if verdict_text in ["AGREE", "DISAGREE"]:
                    verdict = verdict_text
            elif line.startswith("CRITICISM:"):
                criticism = line[len("CRITICISM:"):].strip()
        
        if not verdict:
            raise ValueError("Could not extract VERDICT from response")
        
        # If verdict is DISAGREE, criticism should be present
        if verdict == "DISAGREE" and not criticism:
            logger.warning("Validator disagreed but did not provide criticism")
        
        return ValidatorResponse(verdict=verdict, criticism=criticism)
    
    def validate_answer(
        self,
        question: QuestionItem,
        selected_answer: str,
        reasoning: str
    ) -> ValidatorResponse:
        """
        Validate an answer selection.
        
        Args:
            question: The question being answered
            selected_answer: The answer selected by the answerer
            reasoning: The answerer's reasoning
            
        Returns:
            ValidatorResponse with verdict and optional criticism
            
        Raises:
            Exception: If API call fails or response cannot be parsed
        """
        logger.info(f"Validating answer for question {question.questionNumber}")
        
        # Build the prompt
        prompt = self._build_prompt(question, selected_answer, reasoning)
        logger.debug(f"Validator prompt: {prompt}")
        
        # Get response from Gemini
        response_text = self.gemini_client.generate_response(prompt)
        logger.debug(f"Validator response: {response_text}")
        
        # Parse the response
        try:
            parsed_response = self._parse_response(response_text)
            logger.info(
                f"Question {question.questionNumber}: Validator {parsed_response.verdict}"
            )
            return parsed_response
        except ValueError as e:
            logger.error(f"Failed to parse validator response: {e}")
            raise
