"""
Tools for STAR Answer Generation Pipeline

This module provides minimal tools for tracking history and retrieving outputs.
The orchestrator handles most state management, so these tools are kept simple.
"""

import json
import datetime
from typing import Dict, Any, List # Added List for clarity if needed later
import sys # Added for flushing print statements
import logging

logger = logging.getLogger(__name__)
from google.adk.tools import ToolContext
from .schemas import STARResponse, Critique
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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



def retrieve_final_output_from_state(tool_context: ToolContext) -> str: # Changed return type to str
    logger.info("---- retrieve_final_output_from_state: ENTERED ----")
    full_iteration_history_from_state = tool_context.session.state.get('full_iteration_history', [])
    logger.info(f"[TOOLS LOG] full_iteration_history_from_state received by tool ({len(full_iteration_history_from_state)} items): {str(full_iteration_history_from_state)[:500]}...")
    """
    Retrieve and format the final output for the frontend, using full_iteration_history.

    Args:
        tool_context (ToolContext): The context object providing access to the current state.

    Returns:
        str: A JSON string representing the final answer and history.
             Returns a JSON string with an error message if history is not found or is invalid.
    """
    if not isinstance(full_iteration_history_from_state, list):
        logger.error(f"[TOOLS LOG] Invalid full_iteration_history_from_state type: {type(full_iteration_history_from_state)}. Expected list.")
        return json.dumps({"error": "Invalid history format: not a list"})

    if not full_iteration_history_from_state:
        logger.warning("[TOOLS LOG] full_iteration_history_from_state is empty.")
        # Attempt to retrieve the latest answer directly from state if history is empty
        latest_answer = tool_context.state.get('latest_star_answer', None)
        latest_rating = tool_context.state.get('latest_rating', None)
        if latest_answer and latest_rating is not None:
            logger.info("[TOOLS LOG] Found latest_answer and latest_rating in state as fallback for empty history.")
            if isinstance(latest_answer, str):
                try:
                    latest_answer = json.loads(latest_answer)
                except json.JSONDecodeError as e:
                    logger.error(f"[TOOLS LOG] Error decoding latest_answer string (fallback): {e}")
                    return json.dumps({"error": "Failed to decode latest_answer string (fallback)."})
            
            return json.dumps({
                "answer": latest_answer,
                "history": [],
                "rating": float(latest_rating) # Ensure rating is float
            }, cls=NpEncoder)
        else:
            logger.error("[TOOLS LOG] No full_iteration_history and no fallback latest_answer/rating found for empty history.")
            return json.dumps({"error": "No history or answer found in state for empty history"})

    formatted_history = []
    final_answer_candidate = None
    final_rating_candidate = 0.0 # Default to float

    for item in full_iteration_history_from_state:
        logger.debug(f"[TOOLS LOG] Processing item: {item}")
        iteration_entry = {}
        if not isinstance(item, dict):
            logger.warning(f"[TOOLS LOG] Skipping non-dict item in history: {item}")
            continue

        iteration_entry['iteration_number'] = item.get('iteration_number', 'N/A')
        
        answer_data = item.get('answer')
        if isinstance(answer_data, str):
            try:
                iteration_entry['answer'] = json.loads(answer_data)
            except json.JSONDecodeError:
                logger.error(f"[TOOLS LOG] Failed to parse answer string in iteration {iteration_entry.get('iteration_number', 'N/A')}: {answer_data}")
                iteration_entry['answer'] = {"error": "Malformed answer string", "original_string": answer_data}
        elif isinstance(answer_data, dict):
            iteration_entry['answer'] = answer_data
        else:
            iteration_entry['answer'] = {"error": "Answer not found or invalid type"}
            logger.warning(f"[TOOLS LOG] Answer not found or invalid type for iteration {iteration_entry.get('iteration_number', 'N/A')}. Type: {type(answer_data)}")

        critique_data = item.get('critique')
        parsed_critique_rating = 0.0 # Default rating from critique
        if isinstance(critique_data, str):
            try:
                iteration_entry['critique'] = json.loads(critique_data)
                if isinstance(iteration_entry['critique'], dict):
                    raw_crit_rating = iteration_entry['critique'].get('rating')
                    if raw_crit_rating is not None:
                        try: parsed_critique_rating = float(raw_crit_rating)
                        except (ValueError, TypeError): logger.warning(f"[TOOLS LOG] Malformed rating in parsed critique string: {raw_crit_rating}")
            except json.JSONDecodeError:
                logger.error(f"[TOOLS LOG] Failed to parse critique string in iteration {iteration_entry.get('iteration_number', 'N/A')}: {critique_data}")
                iteration_entry['critique'] = {"error": "Malformed critique string", "original_string": critique_data}
        elif isinstance(critique_data, dict):
            iteration_entry['critique'] = critique_data
            raw_crit_rating = critique_data.get('rating')
            if raw_crit_rating is not None:
                try: parsed_critique_rating = float(raw_crit_rating)
                except (ValueError, TypeError): logger.warning(f"[TOOLS LOG] Malformed rating in critique dict: {raw_crit_rating}")
        else:
            iteration_entry['critique'] = {"error": "Critique not found or invalid type"}
            logger.warning(f"[TOOLS LOG] Critique not found or invalid type for iteration {iteration_entry.get('iteration_number', 'N/A')}. Type: {type(critique_data)}")

        # Determine overall rating for the iteration
        iter_rating_raw = item.get('rating', parsed_critique_rating) # Prioritize top-level rating, fallback to critique's rating
        try:
            iter_rating = float(iter_rating_raw)
        except (ValueError, TypeError):
            logger.warning(f"[TOOLS LOG] Could not parse iteration rating '{iter_rating_raw}', defaulting to 0.0 for iteration {iteration_entry.get('iteration_number', 'N/A')}.")
            iter_rating = 0.0
        
        iteration_entry['rating'] = iter_rating
        # Ensure critique also reflects this definitive iteration rating
        if isinstance(iteration_entry['critique'], dict):
            iteration_entry['critique']['rating'] = iter_rating 

        if iteration_entry['answer'] and not iteration_entry['answer'].get('error'):
            final_answer_candidate = iteration_entry['answer'] # Keep updating with last good answer
        if iter_rating >= final_rating_candidate: # Update if this iteration has a higher or equal rating
            final_rating_candidate = iter_rating
            # final_answer_candidate = iteration_entry['answer'] # This was an old logic, now final_answer is just last good one

        formatted_history.append(iteration_entry)

    # Final answer is the 'latest_star_answer' from state, or the last good one processed from history
    timing_data_from_state = tool_context.session.state.get('timing_data', {})
    final_star_answer = tool_context.session.state.get('latest_star_answer', final_answer_candidate)
    if isinstance(final_star_answer, str):
        try:
            final_star_answer = json.loads(final_star_answer)
        except json.JSONDecodeError as e:
            logger.error(f"[TOOLS LOG] Error decoding final_star_answer from state: {e}. Using last candidate from history if available.")
            final_star_answer = final_answer_candidate if final_answer_candidate else {"error": "Failed to decode latest_star_answer from state and no history candidate."}
    elif not final_star_answer and final_answer_candidate: # If state didn't have it, but history processing did
        final_star_answer = final_answer_candidate
    elif not final_star_answer: # If still no answer
        logger.error("[TOOLS LOG] No definitive final answer could be determined.")
        final_star_answer = {"error": "No definitive answer found."}

    # Final rating is the 'latest_rating' from state, or the highest one processed from history
    overall_final_rating = tool_context.session.state.get('latest_rating', final_rating_candidate)
    try:
        overall_final_rating = float(overall_final_rating)
    except (ValueError, TypeError):
        logger.warning(f"[TOOLS LOG] Could not parse overall_final_rating '{overall_final_rating}', using final_rating_candidate {final_rating_candidate}.")
        overall_final_rating = float(final_rating_candidate) # Fallback to history's best

    output_payload = {
        "answer": final_star_answer, 
        "history": formatted_history,
        "rating": overall_final_rating
    }
    logger.info(f"[TOOLS LOG] Successfully processed history. Final payload for frontend snippet: {str(output_payload)[:500]}...")
    final_json_string = json.dumps(output_payload, cls=NpEncoder, indent=2)
    logger.info(f"[TOOLS LOG] Full JSON string being returned by retrieve_final_output_from_state (len: {len(final_json_string)}). Snippet: {final_json_string[:1000]}...")
    logger.info(f"[TOOLS LOG] Returning JSON (len: {len(final_json_string)}): {final_json_string[:300]}...")
    return final_json_string