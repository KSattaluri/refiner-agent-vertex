from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


class STARResponse(BaseModel):
    """Schema for STAR format interview responses"""

    situation: Optional[str] = Field(
        default=None,
        description="Description of the situation or context where the experience occurred"
    )

    task: Optional[str] = Field(
        default=None,
        description="Explanation of the specific task, responsibility, or challenge faced"
    )

    action: Optional[str] = Field(
        default=None,
        description="Detailed description of the specific actions taken to address the situation"
    )

    result: Optional[str] = Field(
        default=None,
        description="The outcomes, accomplishments, or lessons learned from the experience"
    )


class Critique(BaseModel):
    """Schema for STAR answer critiques"""

    rating: float = Field(
        0.0,
        ge=0.0,
        le=5.0,
        description="Overall numerical rating (0.0-5.0), e.g., 4.7"
    )
    structure_feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the answer's structure"
    )
    relevance_feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the answer's relevance"
    )
    specificity_feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the answer's specificity"
    )
    professional_impact_feedback: Optional[str] = Field(
        default=None,
        description="Feedback on the answer's professional impact"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of concrete suggestions for improvement"
    )
    raw_critique_text: Optional[str] = Field(
        default=None,
        description="The raw text output from the critique agent, if direct JSON parsing fails or for debugging"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Qualitative feedback or suggestions for improvement based on the criteria."
    )


class StarAnswerAndCritique(BaseModel):
    """Schema for pairing a STAR response with its critique"""

    answer: Optional[STARResponse] = Field(
        default=None,
        description="The STAR format answer being critiqued"
    )
    critique: Critique = Field(
        description="The critique of the STAR answer"
    )
    iteration: int = Field(
        description="Which refinement iteration this represents (starting at 1)"
    )


class RequestDetails(BaseModel):
    """Schema for the original user request parameters"""

    role: str = Field(
        description="The job role or position being applied for"
    )
    industry: str = Field(
        description="The industry or sector of the job"
    )
    question: str = Field(
        description="The interview question to answer in STAR format"
    )
    resume: Optional[str] = Field(
        default="",
        description="Optional resume content to personalize the answer"
    )
    job_description: Optional[str] = Field(
        default="",
        description="Optional job description to tailor the answer"
    )


class CritiqueDetails(BaseModel):
    """Simplified critique details for the front-end"""

    rating: float = Field(
        0.0,
        ge=0.0,
        le=5.0,
        description="Overall numerical rating (0.0-5.0), e.g., 4.7"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of concrete suggestions for improvement"
    )


class IterationRecord(BaseModel):
    """Schema for recording a complete iteration in the refinement process"""

    iteration_id: int = Field(
        description="The sequential number of this iteration (1 for initial, 2+ for refinements)"
    )
    star_response: Optional[STARResponse] = Field(
        default=None,
        description="The STAR format answer generated for this iteration"
    )
    critique_details: CritiqueDetails = Field(
        description="The critique's rating and suggestions for this iteration"
    )
    timestamp_utc: str = Field(
        description="UTC timestamp when this iteration was created, ISO 8601 format"
    )


class EnhancedAgentFinalOutput(BaseModel):
    """Enhanced schema for the final output from the agent, optimized for front-end use"""

    request_details: RequestDetails = Field(
        description="The original request parameters from the user"
    )
    interaction_history: List[IterationRecord] = Field(
        description="Complete history of all iterations with their responses and critiques"
    )
    final_status: str = Field(
        description="Status code indicating completion reason: COMPLETED_HIGH_RATING, COMPLETED_MAX_ITERATIONS, ERROR_AGENT_PROCESSING"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed error message if final_status indicates an error"
    )


class AgentFinalOutput(BaseModel):
    """Schema for the final output from the agent, including history."""

    final_star_answer: Optional[STARResponse] = Field(
        default=None,
        description="The final version of the STAR answer after all refinements"
    )
    all_iterations: List[StarAnswerAndCritique] = Field(
        default_factory=list,
        description="History of all iterations with their paired critiques"
    )
    highest_rated_iteration: int = Field(
        default=0,
        description="Which iteration had the highest rating (0-based index into all_iterations)"
    )
    final_rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="The final rating achieved for the STAR answer"
    )

    # Keep these for backward compatibility, can be removed later
    answer_history: Optional[List[Union[STARResponse, Dict[str, Any], str]]] = Field(
        default_factory=list,
        description="DEPRECATED: Use all_iterations instead. History of all STAR answers."
    )
    critique_history: Optional[List[Union[Critique, Dict[str, Any], str]]] = Field(
        default_factory=list,
        description="DEPRECATED: Use all_iterations instead. History of all critiques."
    )