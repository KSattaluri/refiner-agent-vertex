"""
Tools for STAR Answer Generation Pipeline

This module provides minimal tools for tracking history and retrieving outputs.
The orchestrator handles most state management, so these tools are kept simple.
"""

import json
import datetime
from typing import Dict, Any
from google.adk.tools import ToolContext
from .schemas import STARResponse, Critique


def initialize_history(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Initialize empty history tracking structures.
    
    Args:
        tool_context: Context for accessing session state
        
    Returns:
        Success status message
    """
    # Initialize the state with empty lists
    state_delta = {
        "iterations": [],  # Main tracking structure
        "current_iteration": 0,  # Will be set to 1 by orchestrator before first STAR generation
        "highest_rated_iteration": 0,
        "highest_rating": 0.0,
        "final_status": "IN_PROGRESS"
    }
    
    # Apply state updates
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        tool_context.actions.state_delta = state_delta
    else:
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value
    
    return {
        "status": "success",
        "message": "History tracking initialized"
    }


def append_star_response(tool_context: ToolContext, input_key: str = "current_answer") -> Dict[str, Any]:
    """
    Append a STAR response to the iteration history.
    
    Args:
        tool_context: Context for accessing session state
        input_key: Key containing the STAR response
        
    Returns:
        Success status message
    """
    # Get the response from state
    response = tool_context.state.get(input_key)
    if not response:
        return {"status": "error", "message": "No response to append"}
    
    # Get the current iteration number from state (set by orchestrator)
    current_iter = tool_context.state.get("current_iteration", 1)
    
    # Parse response if it's a string
    answer_obj = response
    if isinstance(response, str):
        try:
            # Clean markdown formatting
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("\n", 1)[0]
            answer_obj = json.loads(cleaned)
        except:
            # Fallback to string
            answer_obj = {"situation": response[:200], "task": "", "action": "", "result": ""}
    
    # Create or update iteration
    iterations = tool_context.state.get("iterations", [])
    
    # Find existing iteration or create new one
    existing = None
    for i, iter_data in enumerate(iterations):
        if iter_data.get("iteration") == current_iter:
            existing = i
            break
    
    iteration_data = {
        "iteration": current_iter,
        "answer": answer_obj,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    if existing is not None:
        iterations[existing].update(iteration_data)
    else:
        iterations.append(iteration_data)
    
    # Update state (don't change current_iteration, that's managed by orchestrator)
    state_delta = {
        "iterations": iterations,
        "current_answer": answer_obj
    }
    
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        tool_context.actions.state_delta = state_delta
    else:
        for key, value in state_delta.items():
            tool_context.state[key] = value
    
    return {
        "status": "success",
        "message": f"Added response to iteration {current_iter}"
    }


def rate_star_answer(answer: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Simple acknowledgment that answer was received for rating.
    The actual rating is done by the LLM agent.
    
    Args:
        answer: The STAR answer to rate
        tool_context: Context for accessing session state
        
    Returns:
        Acknowledgment message
    """
    tool_context.state["evaluation_performed"] = True
    
    return {
        "message": "Answer received for evaluation. Please provide rating and feedback."
    }


def append_critique(tool_context: ToolContext, input_key: str = "critique_feedback") -> Dict[str, Any]:
    """
    Append critique to the current iteration.
    
    Args:
        tool_context: Context for accessing session state
        input_key: Key containing the critique
        
    Returns:
        Success status message
    """
    pass


def retrieve_final_output_from_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieve and format the final output for the frontend, using full_iteration_history.

    Args:
        tool_context: Context for accessing session state

    Returns:
        Formatted output with history and metadata
    """
    # Get the new history list populated by the orchestrator
    # Each item is: {"iteration_number": ..., "answer": {...}, "critique": {...}, "rating": ...}
    full_iteration_history_from_state = tool_context.state.get("full_iteration_history", [])
    
    print(f"[TOOLS DEBUG] Found {len(full_iteration_history_from_state)} items in full_iteration_history_from_state")

    final_answer_to_display = None  # The STAR answer from the highest-rated iteration
    highest_rating_achieved = 0.0
    
    # This will be the 'history' array in the final JSON sent to the frontend
    # Each item will have: role, industry, question, answer (STAR object), critique (critique object)
    history_for_frontend = []

    if not full_iteration_history_from_state:
        # Handle cases where there's no iteration history (e.g., error before first generation)
        current_ans_direct = tool_context.state.get("current_answer")
        if isinstance(current_ans_direct, dict):
            final_answer_to_display = current_ans_direct
        elif current_ans_direct: # If it's a string or something else
            final_answer_to_display = {"situation": str(current_ans_direct)[:300], "task": "N/A", "action": "N/A", "result": "N/A"}
        else: # Default if no answer at all
            final_answer_to_display = {"situation": "Not available", "task": "Not available", "action": "Not available", "result": "Not available"}
        # highest_rating_achieved remains 0.0

    else: # Process the full_iteration_history_from_state
        for iteration_data in full_iteration_history_from_state:
            iter_answer = iteration_data.get("answer", {})
            if not isinstance(iter_answer, dict): 
                iter_answer = {"situation": "Answer data malformed", "task": "", "action": "", "result": ""}

            iter_critique = iteration_data.get("critique", {})
            if not isinstance(iter_critique, dict): 
                iter_critique = {"rating": 0.0, "suggestions": ["Critique data malformed"]}
            
            # Ensure rating within critique is present, using the top-level 'rating' from iteration_data as primary
            iter_rating = float(iteration_data.get("rating", iter_critique.get("rating", 0.0)))
            if "rating" not in iter_critique: # Add rating to critique if not already there from parsing
                 iter_critique["rating"] = iter_rating

            history_item = {
                "role": tool_context.state.get("role", "N/A"),
                "industry": tool_context.state.get("industry", "N/A"),
                "question": tool_context.state.get("question", "N/A"),
                "answer": iter_answer,     
                "critique": iter_critique  
            }
            history_for_frontend.append(history_item)

            if iter_rating >= highest_rating_achieved:
                highest_rating_achieved = iter_rating
                final_answer_to_display = iter_answer
        
        if not final_answer_to_display and history_for_frontend: # Fallback if all ratings were 0
            final_answer_to_display = history_for_frontend[-1]["answer"]

    if final_answer_to_display is None: # Should only happen if history was empty AND no current_answer
        final_answer_to_display = {"situation": "No answer available", "task": "", "action": "", "result": ""}

    output_payload = {
        "answer": final_answer_to_display,
        "history": history_for_frontend,
        "rating": highest_rating_achieved 
    }
    
    timing_data = tool_context.state.get("timing_data")
    if timing_data:
        print(f"[TOOLS DEBUG] Timing data found in state: {timing_data}")
    else:
        print(f"[TOOLS DEBUG] No timing data found in state")

    print(f"[TOOLS DEBUG] History built for frontend output: {len(history_for_frontend)}")
    if history_for_frontend:
        first_hist_item = history_for_frontend[0]
        print(f"[TOOLS DEBUG] First history item for frontend (keys): {list(first_hist_item.keys())}")
        print(f"[TOOLS DEBUG] First history item answer (keys): {list(first_hist_item.get('answer', {}).keys())}")
        print(f"[TOOLS DEBUG] First history item critique (keys): {list(first_hist_item.get('critique', {}).keys())}")

    result = {
        "status": "success",
        "message": f"Retrieved output with {len(history_for_frontend)} history items.",
        "retrieved_output": output_payload 
    }

    print(f"[TOOLS DEBUG] Final result being returned by tool (keys): {list(result.keys())}")
    print(f"[TOOLS DEBUG] Final 'retrieved_output' (keys): {list(result['retrieved_output'].keys())}")
    print(f"[TOOLS DEBUG] Final 'retrieved_output' history length: {len(result['retrieved_output'].get('history', []))}")
    
    return result