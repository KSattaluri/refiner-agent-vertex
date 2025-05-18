"""
Input collection tool for the STAR Answer Generator
"""

from typing import Dict, Any, Optional
from google.adk.tools import ToolContext


def collect_star_inputs(
    role: str,
    industry: str, 
    question: str,
    resume: str = "",
    job_description: str = "",
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Collect and validate the inputs needed for STAR answer generation.
    
    Args:
        role: The job position being applied for
        industry: The industry or sector of the job
        question: The interview question to answer
        resume: Optional resume content
        job_description: Optional job description
        tool_context: Optional tool context for state management
        
    Returns:
        Dictionary containing the collected inputs
    """
    # Validate required fields
    if not role or not industry or not question:
        missing = []
        if not role: missing.append("role")
        if not industry: missing.append("industry") 
        if not question: missing.append("question")
        
        return {
            "status": "error",
            "message": f"Missing required fields: {', '.join(missing)}",
            "missing_fields": missing
        }
    
    # Store input details in state if context provided
    if tool_context:
        state_delta = {
            "role": role,
            "industry": industry,
            "question": question,
            "resume": resume,
            "job_description": job_description,
            "request_details": {
                "role": role,
                "industry": industry,
                "question": question,
                "resume": resume,
                "job_description": job_description
            }
        }
        
        # Apply state update
        if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
            tool_context.actions.state_delta = state_delta
        else:
            for key, value in state_delta.items():
                tool_context.state[key] = value
    
    return {
        "status": "success",
        "role": role,
        "industry": industry,
        "question": question,
        "resume": resume,
        "job_description": job_description
    }