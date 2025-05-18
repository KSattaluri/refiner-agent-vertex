"""
Validation models for the STAR Answer Generator API.

This module defines Pydantic models for validating API requests and responses,
ensuring data integrity and providing clear error messages for malformed data.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, constr
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

# Request Models

class STARGeneratorRequest(BaseModel):
    """
    Validates incoming requests to the STAR Answer Generator API.
    
    Enforces constraints on fields like minimum length and format requirements.
    """
    role: constr(min_length=2, max_length=100) = Field(
        ..., 
        description="Job role or position being applied for"
    )
    industry: constr(min_length=2, max_length=100) = Field(
        ..., 
        description="Industry or sector of the job"
    )
    question: constr(min_length=10) = Field(
        ..., 
        description="Interview question to answer in STAR format"
    )
    resume: str = Field(
        "", 
        description="Optional resume text to reference in the answer"
    )
    job_description: str = Field(
        "", 
        description="Optional job description to tailor the answer to"
    )
    
    @field_validator('role', 'industry')
    @classmethod
    def check_valid_values(cls, v):
        """Ensures fields contain meaningful text, not just spaces or special characters."""
        if v.strip() == "" or not any(c.isalpha() for c in v):
            raise ValueError(f"Must contain alphabetic characters, not just spaces or special characters")
        return v

    @field_validator('question')
    @classmethod
    def check_question_format(cls, v):
        """Validates that the question is properly formatted."""
        if not any(v.endswith(end) for end in ['?', '.', '!']):
            raise ValueError("Question must end with proper punctuation (?, ., or !)")
        if len(v.split()) < 3:
            raise ValueError("Question must be at least 3 words long")
        return v

# Response Models

class STARAnswer(BaseModel):
    """Validates the structure of a STAR format answer."""
    situation: Optional[str] = Field(None, description="Description of the situation or context")
    task: Optional[str] = Field(None, description="Explanation of the specific task or challenge")
    action: Optional[str] = Field(None, description="Description of the actions taken")
    result: Optional[str] = Field(None, description="Outcomes and accomplishments achieved")

    # Removed validator to allow empty objects, as these appear in the history

class CritiqueFeedback(BaseModel):
    """Validates the structure of critique feedback."""
    rating: float = Field(..., ge=0.0, le=5.0, description="Numerical rating from 0.0 to 5.0")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    structure_feedback: Optional[str] = Field(None, description="Feedback on structure")
    relevance_feedback: Optional[str] = Field(None, description="Feedback on relevance")
    specificity_feedback: Optional[str] = Field(None, description="Feedback on specificity")
    professional_impact_feedback: Optional[str] = Field(None, description="Feedback on professional impact")

class HistoryItem(BaseModel):
    """Validates a single item in the history of STAR answers and critiques."""
    iteration: int = Field(..., ge=1, description="Iteration number, starting from 1")
    star_answer: Optional[STARAnswer] = Field(None, description="STAR format answer for this iteration")
    critique: CritiqueFeedback = Field(..., description="Critique feedback for this iteration")
    timestamp: Optional[str] = Field(None, description="Timestamp when this iteration was created")

class Metadata(BaseModel):
    """Validates metadata about the response."""
    status: str = Field(..., description="Status code: COMPLETED, COMPLETED_HIGH_RATING, ERROR_*")
    highest_rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Highest rating achieved")
    role: Optional[str] = Field(None, description="Job role from the request")
    industry: Optional[str] = Field(None, description="Industry from the request")
    question: Optional[str] = Field(None, description="Question from the request")
    error_message: Optional[str] = Field(None, description="Error message if status indicates an error")

class STARGeneratorResponse(BaseModel):
    """
    Validates the complete response structure from the STAR Answer Generator API.

    This ensures the API returns a consistent structure even in error cases.
    """
    star_answer: Optional[STARAnswer] = Field(
        None,
        description="Final STAR format answer after all refinements"
    )
    feedback: Optional[CritiqueFeedback] = Field(
        None,
        description="Feedback on the final answer"
    )
    history: List[HistoryItem] = Field(
        default_factory=list,
        description="History of all iterations with their answers and critiques"
    )
    metadata: Metadata = Field(
        ...,
        description="Metadata about the request and response"
    )

# LLM Input/Output Models

class LLMPromptData(BaseModel):
    """
    Validates and sanitizes data before sending to LLM to prevent prompt injection.
    
    This model is used internally and not exposed directly in the API.
    """
    role: str
    industry: str
    question: str
    
    @field_validator('*')
    @classmethod
    def sanitize_inputs(cls, v):
        """Removes potential prompt injection attempts."""
        dangerous_patterns = [
            "ignore previous instructions",
            "disregard",
            "system prompt",
            "ignore the above",
            "ignore all instructions",
            "as an AI",
            "as an LLM"
        ]
        for pattern in dangerous_patterns:
            v = v.replace(pattern, "[filtered]")
        return v

# Error Models

class ValidationError(BaseModel):
    """Model for validation error responses."""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Error message explaining the validation failure")

class ErrorResponse(BaseModel):
    """Standardized error response structure."""
    star_answer: Optional[Dict[str, Any]] = Field(None, description="Empty in error responses")
    feedback: Optional[Dict[str, Any]] = Field(None, description="Empty in error responses")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Empty or partial in error responses")
    metadata: Dict[str, Any] = Field(
        ...,
        description="Error metadata including status and error message"
    )
    validation_errors: Optional[List[ValidationError]] = Field(
        None,
        description="List of validation errors if applicable"
    )