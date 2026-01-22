"""FastAPI server for AI QA Validator API."""

import logging
import os
import uuid
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from models.schemas import QuestionItem, AnswerResult, SimpleAnswerResult, SystemConfig
from workers.question_processor import QuestionProcessor
from agents.multi_agent_validator import MultiAgentValidator
from agents.answerer_agent import AnswererAgent
from agents.validator_agent import ValidatorAgent
from agents.gemini_client import GeminiClient
from utils.logging_config import configure_logging, set_request_id


logger = logging.getLogger(__name__)


# Global instances
question_processor: QuestionProcessor = None
system_config: SystemConfig = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events.
    """
    # Startup: Initialize system components
    global question_processor, system_config
    
    # Load configuration from environment
    try:
        system_config = SystemConfig(
            geminiApiKey=os.getenv("GEMINI_API_KEY", ""),
            apiPort=int(os.getenv("API_PORT", 8000)),
            maxConcurrentWorkers=int(os.getenv("MAX_CONCURRENT_WORKERS", 5)),
            maxValidationIterations=int(os.getenv("MAX_VALIDATION_ITERATIONS", 5)),
            geminiMaxRetries=int(os.getenv("GEMINI_MAX_RETRIES", 3)),
            geminiBaseRetryDelayMs=int(os.getenv("GEMINI_BASE_RETRY_DELAY_MS", 1000)),
            logLevel=os.getenv("LOG_LEVEL", "INFO")
        )
        
        # Configure structured logging
        configure_logging(system_config.logLevel)
        
        logger.info("Starting AI QA Validator API...")
        
        # Validate API key is present
        if not system_config.geminiApiKey:
            logger.error("GEMINI_API_KEY environment variable is not set")
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        logger.info(f"Configuration loaded: port={system_config.apiPort}, "
                   f"workers={system_config.maxConcurrentWorkers}, "
                   f"max_iterations={system_config.maxValidationIterations}")
        
        # Initialize Gemini client
        gemini_client = GeminiClient(
            api_key=system_config.geminiApiKey,
            max_retries=system_config.geminiMaxRetries,
            base_retry_delay_ms=system_config.geminiBaseRetryDelayMs
        )
        
        # Initialize agents
        answerer_agent = AnswererAgent(gemini_client)
        validator_agent = ValidatorAgent(gemini_client)
        
        # Initialize multi-agent validator
        multi_agent_validator = MultiAgentValidator(
            answerer_agent=answerer_agent,
            validator_agent=validator_agent,
            max_iterations=system_config.maxValidationIterations
        )
        
        # Initialize question processor
        question_processor = QuestionProcessor(
            validator=multi_agent_validator,
            max_concurrent_workers=system_config.maxConcurrentWorkers
        )
        
        logger.info("AI QA Validator API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI QA Validator API...")


# Create FastAPI application
app = FastAPI(
    title="AI QA Validator API",
    description="Multi-agent AI system for validating question answers",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Middleware to add request ID for tracing.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Set request ID in logging context
    set_request_id(request_id)
    
    # Log incoming request
    logger.info(f"{request.method} {request.url.path}")
    
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code}")
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    set_request_id(request_id)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "details": "An unexpected error occurred while processing your request",
            "request_id": request_id
        }
    )


@app.post("/api/answer-questions", response_model=List[SimpleAnswerResult])
async def answer_questions(
    questions: List[QuestionItem],
    request: Request
) -> List[SimpleAnswerResult]:
    """
    Process a list of multiple-choice questions through multi-agent validation.
    
    Args:
        questions: List of QuestionItem objects to process
        request: FastAPI request object (for request ID)
        
    Returns:
        List of SimpleAnswerResult objects with questionNumber and selectedAnswer only
        
    Raises:
        HTTPException: 400 for invalid input, 503 for service unavailable
    """
    request_id = getattr(request.state, "request_id", "unknown")
    set_request_id(request_id)
    
    logger.info(f"Received request to process {len(questions)} questions")
    
    # Validate that we have questions
    if not questions:
        logger.warning("Empty question list received")
        return []
    
    # Log request details
    logger.debug(f"Questions: {[q.questionNumber for q in questions]}")
    
    try:
        # Check if question processor is initialized
        if question_processor is None:
            logger.error("Question processor not initialized")
            raise HTTPException(
                status_code=503,
                detail="Service is not ready. Please try again later."
            )
        
        # Process questions through the multi-agent validator
        results = await question_processor.process_questions(questions)
        
        # Convert to simplified response format (only questionNumber and selectedAnswer)
        simple_results = []
        for result in results:
            if result.error:
                # For errors, we could either skip them or raise an exception
                # Let's raise an exception for failed questions
                logger.error(f"Question {result.questionNumber} failed: {result.error}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Question {result.questionNumber} failed: {result.error}"
                )
            
            simple_results.append(
                SimpleAnswerResult(
                    questionNumber=result.questionNumber,
                    selectedAnswer=result.selectedAnswer
                )
            )
        
        # Log response summary
        logger.info(f"Completed processing {len(simple_results)} questions successfully")
        
        return simple_results
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(
            f"Error processing questions: {e}",
            exc_info=True
        )
        
        # Check if it's a Gemini API availability issue
        if "API" in str(e) or "connection" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again later."
            )
        
        # Re-raise as internal server error
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process questions: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status information about the service
    """
    is_ready = question_processor is not None and system_config is not None
    
    return {
        "status": "healthy" if is_ready else "starting",
        "ready": is_ready,
        "version": "1.0.0"
    }
