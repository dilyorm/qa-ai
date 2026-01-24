"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class AnswerOption(BaseModel):
    """Model for a single answer option."""
    content: str
    isRight: bool = False

class QuestionItem(BaseModel):
    """Model for a multiple-choice question item."""
    
    content: str = Field(..., min_length=1, description="Full question text with context")
    title: str = Field(..., min_length=1, description="Brief question summary")
    type: str = Field(..., description="Question type (e.g., 'option')")
    column: Optional[str] = Field(None, description="Layout column type")
    answers: List[AnswerOption] = Field(..., min_length=2, description="Array of possible answers")
    questionNumber: str = Field(..., min_length=1, description="Unique identifier for question")
    
    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v: List[AnswerOption]) -> List[AnswerOption]:
        """Ensure all answer options are valid."""
        if not v:
            raise ValueError("Answer list cannot be empty")
        if len(v) < 2:
            raise ValueError("Must have at least 2 answer options")
        for answer in v:
            if not answer.content or not answer.content.strip():
                raise ValueError("Answer content cannot be empty strings")
        return v


class AnswerResult(BaseModel):
    """Model for a question answer result."""
    
    questionNumber: str = Field(..., description="Matches input question number")
    selectedAnswer: Optional[str] = Field(None, description="The selected answer letter (A, B, C, D, etc.)")
    error: Optional[str] = Field(None, description="Error message (if failed)")
    validationIterations: int = Field(..., ge=0, description="Number of agent iterations")
    processingTimeMs: int = Field(..., ge=0, description="Total processing time in milliseconds")
    
    @field_validator('selectedAnswer', 'error')
    @classmethod
    def validate_result_state(cls, v, info):
        """Ensure either selectedAnswer or error is present, but not both."""
        # This validation happens after all fields are set
        return v


class SimpleAnswerResult(BaseModel):
    """Simplified model for question answer result (API response)."""
    
    questionNumber: str = Field(..., description="Matches input question number")
    selectedAnswer: str = Field(..., description="The selected answer letter (A, B, C, D, etc.)")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "questionNumber": "1",
                "selectedAnswer": "B"
            }
        }


class SystemConfig(BaseModel):
    """Model for system configuration from environment variables."""
    
    geminiApiKey: str = Field(..., min_length=1, description="Gemini API authentication key")
    apiPort: int = Field(default=8000, ge=1, le=65535, description="API server port")
    maxConcurrentWorkers: int = Field(default=5, ge=1, description="Maximum concurrent workers")
    maxValidationIterations: int = Field(default=5, ge=1, description="Maximum validation loop iterations")
    geminiMaxRetries: int = Field(default=3, ge=1, description="Maximum Gemini API retry attempts")
    geminiBaseRetryDelayMs: int = Field(default=1000, ge=0, description="Base retry delay in milliseconds")
    logLevel: str = Field(default="INFO", description="Logging level")
    
    @field_validator('logLevel')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper
