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
        "current_iteration": 0,
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
    
    # Determine iteration number
    current_iter = tool_context.state.get("current_iteration", 0)
    current_iter += 1
    
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
    
    # Update state
    state_delta = {
        "iterations": iterations,
        "current_iteration": current_iter,
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
    # Get critique from state
    critique = tool_context.state.get(input_key)
    if not critique:
        return {"status": "error", "message": "No critique to append"}
    
    # Get current iteration
    current_iter = tool_context.state.get("current_iteration", 1)
    iterations = tool_context.state.get("iterations", [])
    
    # Parse critique if needed
    critique_obj = critique
    if isinstance(critique, str):
        try:
            # Clean markdown formatting
            cleaned = critique.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("\n", 1)[0]
            critique_obj = json.loads(cleaned)
        except:
            critique_obj = {"rating": 0.0, "suggestions": ["Unable to parse critique"]}
    
    # Update the current iteration with critique
    for iteration in iterations:
        if iteration.get("iteration") == current_iter:
            iteration["critique"] = critique_obj
            iteration["rating"] = critique_obj.get("rating", 0.0)
            print(f"[TOOLS DEBUG] Updated iteration {current_iter} with critique")
            print(f"[TOOLS DEBUG] Iteration has answer: {'answer' in iteration}")
            print(f"[TOOLS DEBUG] Iteration keys: {list(iteration.keys())}")
            break
    
    # Build the base state delta
    state_delta = {
        "iterations": iterations,
        "critique_feedback": critique_obj
    }

    # Update highest rating if this is a new high score
    rating = critique_obj.get("rating", 0.0)
    if rating > tool_context.state.get("highest_rating", 0.0):
        state_delta.update({
            "highest_rating": rating,
            "highest_rated_iteration": current_iter
        })
    
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        tool_context.actions.state_delta = state_delta
    else:
        for key, value in state_delta.items():
            tool_context.state[key] = value
    
    return {
        "status": "success",
        "message": f"Added critique to iteration {current_iter}"
    }


def retrieve_final_output_from_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieve and format the final output for the frontend.
    
    Args:
        tool_context: Context for accessing session state
        
    Returns:
        Formatted output with history and metadata
    """
    # Build the output structure
    iterations = tool_context.state.get("iterations", [])

    # Debug logging to see what's in iterations and state
    print(f"[TOOLS DEBUG] ===== TOOL CONTEXT DEBUG =====")
    print(f"[TOOLS DEBUG] tool_context type: {type(tool_context)}")
    print(f"[TOOLS DEBUG] tool_context.state type: {type(tool_context.state)}")
    print(f"[TOOLS DEBUG] Raw iterations: {len(iterations)}")
    print(f"[TOOLS DEBUG] Current state keys: {list(tool_context.state.keys())}")

    if iterations:
        print(f"[TOOLS DEBUG] First iteration type: {type(iterations[0])}")
        print(f"[TOOLS DEBUG] First iteration keys: {list(iterations[0].keys()) if isinstance(iterations[0], dict) else 'Not a dict'}")
        if isinstance(iterations[0], dict):
            print(f"[TOOLS DEBUG] First iteration: {iterations[0]}")

    # Check if there's a session object with events
    if hasattr(tool_context, 'session'):
        print(f"[TOOLS DEBUG] tool_context has session attribute")
        if hasattr(tool_context.session, 'events'):
            print(f"[TOOLS DEBUG] Session has events: {len(tool_context.session.events)}")

    print(f"[TOOLS DEBUG] ===== END DEBUG =====")

    # Format the answer and history
    final_answer = None
    history = []
    final_rating = 0.0

    # Process iterations to build history
    for iter_data in iterations:
        answer = iter_data.get("answer", {})
        critique = iter_data.get("critique", {})

        # Build proper critique object with all fields
        critique_obj = {
            "rating": critique.get("rating", 0.0),
            "suggestions": critique.get("suggestions", [])
        }

        # Include all feedback fields if present
        if "structure_feedback" in critique:
            critique_obj["structure_feedback"] = critique["structure_feedback"]
        if "relevance_feedback" in critique:
            critique_obj["relevance_feedback"] = critique["relevance_feedback"]
        if "specificity_feedback" in critique:
            critique_obj["specificity_feedback"] = critique["specificity_feedback"]
        if "professional_impact_feedback" in critique:
            critique_obj["professional_impact_feedback"] = critique["professional_impact_feedback"]

        history_item = {
            "answer": answer,
            "critique": critique_obj
        }
        history.append(history_item)

        # Track the highest rating
        if critique.get("rating", 0.0) > final_rating:
            final_rating = critique.get("rating", 0.0)
            final_answer = answer
    
    # If no iterations, use current answer
    if not final_answer and tool_context.state.get("current_answer"):
        current = tool_context.state.get("current_answer")
        if isinstance(current, dict):
            final_answer = current
        else:
            final_answer = {"situation": str(current)[:200], "task": "", "action": "", "result": ""}
    
    # Build the response
    output = {
        "answer": final_answer or {"situation": "", "task": "", "action": "", "result": ""},
        "history": history,
        "rating": final_rating
    }

    # Debug logging
    print(f"[TOOLS DEBUG] Iterations from state: {len(iterations)}")
    print(f"[TOOLS DEBUG] History built: {len(history)}")
    if history:
        print(f"[TOOLS DEBUG] First history item keys: {list(history[0].keys())}")
        print(f"[TOOLS DEBUG] Full history item 0: {history[0]}")
        print(f"[TOOLS DEBUG] First iteration from state: {iterations[0] if iterations else 'No iterations'}")

    result = {
        "status": "success",
        "message": f"Retrieved output with {len(history)} iterations",
        "retrieved_output": output
    }

    print(f"[TOOLS DEBUG] Returning result with keys: {list(result.keys())}")
    print(f"[TOOLS DEBUG] retrieved_output keys: {list(result['retrieved_output'].keys())}")
    print(f"[TOOLS DEBUG] retrieved_output history length: {len(result['retrieved_output'].get('history', []))}")
    if result['retrieved_output'].get('history'):
        print(f"[TOOLS DEBUG] First history item in output: {result['retrieved_output']['history'][0]}")

    return result