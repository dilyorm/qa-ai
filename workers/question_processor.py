"""Question processor with concurrent worker pool."""

import asyncio
import logging
import time
from typing import List
from models.schemas import QuestionItem, AnswerResult
from agents.multi_agent_validator import MultiAgentValidator


logger = logging.getLogger(__name__)


class QuestionProcessor:
    """Processes multiple questions concurrently using a worker pool."""
    
    def __init__(self, validator: MultiAgentValidator, max_concurrent_workers: int = 5):
        """
        Initialize the question processor.
        
        Args:
            validator: The multi-agent validator instance
            max_concurrent_workers: Maximum number of concurrent workers (default: 5)
        """
        self.validator = validator
        self.max_concurrent_workers = max_concurrent_workers
        self.semaphore = asyncio.Semaphore(max_concurrent_workers)
        logger.info(f"QuestionProcessor initialized with max_concurrent_workers={max_concurrent_workers}")
    
    async def process_questions(self, questions: List[QuestionItem]) -> List[AnswerResult]:
        """
        Process multiple questions concurrently while maintaining order.
        
        Args:
            questions: List of questions to process
            
        Returns:
            List of AnswerResult objects in the same order as input questions
        """
        logger.info(f"Processing {len(questions)} questions with up to {self.max_concurrent_workers} concurrent workers")
        
        # Create tasks for all questions
        tasks = [self._process_single_question(question) for question in questions]
        
        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert any exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                question_num = questions[i].questionNumber
                logger.error(
                    f"Question {question_num} failed with exception: {result}",
                    extra={'question_number': question_num}
                )
                final_results.append(
                    AnswerResult(
                        questionNumber=question_num,
                        error=str(result),
                        validationIterations=0,
                        processingTimeMs=0
                    )
                )
            else:
                final_results.append(result)
        
        logger.info(f"Completed processing {len(questions)} questions")
        return final_results
    
    async def _process_single_question(self, question: QuestionItem) -> AnswerResult:
        """
        Process a single question through the validation loop.
        
        Uses a semaphore to limit concurrent workers.
        
        Args:
            question: The question to process
            
        Returns:
            AnswerResult with the validated answer or error
        """
        async with self.semaphore:
            logger.debug(
                f"Worker acquired for question {question.questionNumber}",
                extra={'question_number': question.questionNumber}
            )
            start_time = time.time()
            
            try:
                # Run the synchronous validation in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                validation_result = await loop.run_in_executor(
                    None,
                    self.validator.validate_question,
                    question
                )
                
                # Calculate processing time
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Create successful result
                result = AnswerResult(
                    questionNumber=question.questionNumber,
                    selectedAnswer=validation_result.selected_answer,
                    validationIterations=validation_result.iterations,
                    processingTimeMs=processing_time_ms
                )
                
                logger.debug(
                    f"Question {question.questionNumber} completed in {processing_time_ms}ms "
                    f"with {validation_result.iterations} iterations",
                    extra={'question_number': question.questionNumber}
                )
                
                return result
                
            except Exception as e:
                # Calculate processing time even for failures
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                logger.error(
                    f"Question {question.questionNumber} failed after {processing_time_ms}ms: {e}",
                    exc_info=True,
                    extra={'question_number': question.questionNumber}
                )
                
                # Create error result
                return AnswerResult(
                    questionNumber=question.questionNumber,
                    error=str(e),
                    validationIterations=0,
                    processingTimeMs=processing_time_ms
                )
