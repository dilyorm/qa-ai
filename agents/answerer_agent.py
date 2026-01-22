"""Answerer Agent for analyzing questions and selecting answers."""

import logging
from typing import Optional
from agents.gemini_client import GeminiClient
from models.schemas import QuestionItem


logger = logging.getLogger(__name__)


class AnswererResponse:
    """Response from the Answerer Agent."""
    
    def __init__(self, selected_answer: str, reasoning: str):
        """
        Initialize an answerer response.
        
        Args:
            selected_answer: The selected answer from available options
            reasoning: The reasoning for the selection
        """
        self.selected_answer = selected_answer
        self.reasoning = reasoning


class AnswererAgent:
    """Agent responsible for analyzing questions and selecting answers."""
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize the Answerer Agent.
        
        Args:
            gemini_client: The Gemini API client to use for generating responses
        """
        self.gemini_client = gemini_client
        logger.info("AnswererAgent initialized")
    
    def _build_prompt(
        self,
        question: QuestionItem,
        previous_answer: Optional[str] = None,
        criticism: Optional[str] = None
    ) -> str:
        """
        Build the prompt for the answerer agent.
        
        Args:
            question: The question to analyze
            previous_answer: Previous answer selection (if reconsidering)
            criticism: Validator's criticism (if reconsidering)
            
        Returns:
            The formatted prompt string
        """
        # Format answer options as lettered list (A, B, C, D, etc.)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        answer_list = "\n".join([f"{letters[i]}. {answer}" for i, answer in enumerate(question.answer)])
        
        prompt = f"""You are an expert at analyzing multiple-choice questions and selecting the best answer.

Question Content:
{question.content}

Question Title:
{question.title}

Available Answers:
{answer_list}
"""
        
        # Add reconsideration context if this is a retry
        if previous_answer and criticism:
            prompt += f"""
Previous Selection: {previous_answer}
Validator Criticism: {criticism}
Please reconsider your answer based on this feedback.
"""
        
        prompt += """
Task: Select the single best answer from the options above. Provide your selection and brief reasoning.

Format your response as:
SELECTED: [letter only - A, B, C, or D]
REASONING: [your explanation]
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> AnswererResponse:
        """
        Parse the agent's response to extract selected answer and reasoning.
        
        Args:
            response_text: The raw response from the Gemini API
            
        Returns:
            AnswererResponse with selected answer and reasoning
            
        Raises:
            ValueError: If response cannot be parsed
        """
        lines = response_text.strip().split('\n')
        selected_answer = None
        reasoning = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SELECTED:"):
                selected_answer = line[len("SELECTED:"):].strip()
            elif line.startswith("REASONING:"):
                reasoning = line[len("REASONING:"):].strip()
        
        if not selected_answer:
            raise ValueError("Could not extract SELECTED from response")
        if not reasoning:
            raise ValueError("Could not extract REASONING from response")
        
        return AnswererResponse(selected_answer=selected_answer, reasoning=reasoning)
    
    def answer_question(
        self,
        question: QuestionItem,
        previous_answer: Optional[str] = None,
        criticism: Optional[str] = None
    ) -> AnswererResponse:
        """
        Analyze a question and select the best answer.
        
        Args:
            question: The question to analyze
            previous_answer: Previous answer selection (if reconsidering)
            criticism: Validator's criticism (if reconsidering)
            
        Returns:
            AnswererResponse with selected answer and reasoning
            
        Raises:
            Exception: If API call fails or response cannot be parsed
        """
        logger.info(f"Answering question {question.questionNumber}")
        
        # Build the prompt
        prompt = self._build_prompt(question, previous_answer, criticism)
        logger.debug(f"Answerer prompt: {prompt}")
        
        # Get response from Gemini
        response_text = self.gemini_client.generate_response(prompt)
        logger.debug(f"Answerer response: {response_text}")
        
        # Parse the response
        try:
            parsed_response = self._parse_response(response_text)
            logger.info(
                f"Question {question.questionNumber}: Selected '{parsed_response.selected_answer}'"
            )
            return parsed_response
        except ValueError as e:
            logger.error(f"Failed to parse answerer response: {e}")
            raise
