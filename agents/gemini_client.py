"""Gemini API client with retry logic and error handling."""

import logging
import time
from typing import Optional
import google.generativeai as genai


logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for interacting with Google Gemini API with retry logic."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        max_retries: int = 3,
        base_retry_delay_ms: int = 1000,
        retry_multiplier: int = 2
    ):
        """
        Initialize the Gemini API client.
        
        Args:
            api_key: Gemini API authentication key
            model: Model name to use (default: gemini-2.0-flash-exp)
            max_retries: Maximum number of retry attempts (default: 3)
            base_retry_delay_ms: Base delay in milliseconds for retries (default: 1000)
            retry_multiplier: Multiplier for exponential backoff (default: 2)
        """
        self.api_key = api_key
        self.model_name = model
        self.max_retries = max_retries
        self.base_retry_delay_ms = base_retry_delay_ms
        self.retry_multiplier = retry_multiplier
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"GeminiClient initialized with model: {self.model_name}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the Gemini API with retry logic.
        
        Implements exponential backoff retry logic:
        - Attempt 1: immediate
        - Attempt 2: wait base_retry_delay_ms (1000ms = 1s)
        - Attempt 3: wait base_retry_delay_ms * retry_multiplier (2000ms = 2s)
        - Attempt 4: wait base_retry_delay_ms * retry_multiplier^2 (4000ms = 4s)
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The generated response text
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Gemini API call attempt {attempt}/{self.max_retries}")
                
                # Make the API call
                response = self.model.generate_content(prompt)
                
                # Extract text from response
                if response and response.text:
                    logger.debug(f"Gemini API call succeeded on attempt {attempt}")
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Gemini API call failed on attempt {attempt}/{self.max_retries}: {str(e)}"
                )
                
                # If this wasn't the last attempt, wait before retrying
                if attempt < self.max_retries:
                    # Calculate delay: base_delay * (multiplier ^ (attempt - 1))
                    delay_ms = self.base_retry_delay_ms * (self.retry_multiplier ** (attempt - 1))
                    delay_seconds = delay_ms / 1000.0
                    
                    logger.info(f"Retrying in {delay_seconds}s...")
                    time.sleep(delay_seconds)
        
        # All retries exhausted
        error_msg = f"Gemini API call failed after {self.max_retries} attempts"
        logger.error(f"{error_msg}: {str(last_exception)}")
        raise Exception(f"{error_msg}: {str(last_exception)}")
