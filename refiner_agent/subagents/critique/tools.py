"""
Tools for STAR Answer Critique Agent

This module provides tools for analyzing and validating STAR format answers.
"""

from typing import Any, Dict
from google.adk.tools.tool_context import ToolContext


def rate_star_answer(answer: str) -> dict:
    """
    Tool to evaluate a STAR format answer and provide a numerical rating.

    Args:
        answer: The STAR answer to evaluate

    Returns:
        Dictionary containing:
            - rating: numerical rating between 1.0 and 5.0
            - critique: textual critique explaining the rating
            - meets_requirements: boolean indicating if answer meets minimum requirements
    """
    # This function doesn't actually rate the answer
    # The LLM will perform the evaluation and return the rating

    # For debugging
    print("\n----------- RATING TOOL CALLED -----------")
    print(f"Answer length: {len(answer)} characters")
    print("-----------------------------------------\n")

    # The actual rating logic is implemented by the LLM
    # This function just passes through the input to the LLM
    return {
        "message": "Analysis performed. Please provide your rating and feedback."
    }


def finalize_agent_output(rating: float, tool_context: ToolContext) -> dict:
    """
    Create the final agent output package when the STAR answer meets quality requirements.

    This should be called when the rating is 4.6 or higher.

    Args:
        rating: The numerical rating (1.0-5.0) of the STAR answer
        tool_context: Context for tool execution

    Returns:
        Dict containing status information
    """
    print("\n----------- FINALIZING OUTPUT -----------")
    print(f"Rating: {rating}")
    print("------------------------------------------\n")

    # Forward to the main finalize_agent_output function in tools.py
    # Import here to avoid circular imports
    import sys
    import os
    import importlib.util

    # Get the parent directory of this module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

    # Load tools module from relative path
    tools_path = os.path.join(parent_dir, "sample_agent", "tools.py")
    spec = importlib.util.spec_from_file_location("main_tools", tools_path)
    tools_module = importlib.util.module_from_spec(spec)
    sys.modules["main_tools"] = tools_module
    spec.loader.exec_module(tools_module)

    # Call the main finalize_agent_output function
    return tools_module.finalize_agent_output(rating, tool_context)