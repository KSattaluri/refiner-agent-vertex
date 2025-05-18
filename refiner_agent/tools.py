"""
Tools for STAR Answer Generation Pipeline

This module provides tools for generating and evaluating STAR format answers,
as well as managing the history of STAR responses and critiques.
"""

import sys
import os
import importlib.util

import json
import datetime
import time
from typing import Dict, Any, List, Optional
from google.adk.tools import ToolContext
from google.adk.events import Event, EventActions
from .schemas import STARResponse, Critique, AgentFinalOutput, EnhancedAgentFinalOutput, RequestDetails, IterationRecord, CritiqueDetails


def generate_star_answer(
    role: str,
    industry: str,
    question: str,
    resume: str = "",
    job_description: str = "",
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Collect information needed to generate a STAR format answer.

    Args:
        role: The job role or position being applied for
        industry: The industry or sector of the job
        question: The interview question to answer in STAR format
        resume: Optional resume content to personalize the answer
        job_description: Optional job description to tailor the answer
        tool_context: Optional context for tool execution, to store state

    Returns:
        Dict with the input parameters that were collected
    """
    # Validation to ensure required fields are present and non-empty
    if not role or not industry or not question:
        missing = []
        if not role: missing.append("role")
        if not industry: missing.append("industry")
        if not question: missing.append("question")

        # Update state with error status if tool_context provided
        if tool_context:
            tool_context.state["final_status"] = "ERROR_INPUT_VALIDATION"
            tool_context.state["error_message"] = f"Missing required information: {', '.join(missing)}"

        return {
            "status": "error",
            "error_message": f"Missing required information: {', '.join(missing)}. Please provide all required fields.",
            "missing_fields": missing
        }

    print("\n----------- INFORMATION COLLECTED -----------")
    print(f"Role: {role}")
    print(f"Industry: {industry}")
    print(f"Question: {question}")
    print(f"Resume provided: {'Yes' if resume else 'No'}")
    print(f"Job description provided: {'Yes' if job_description else 'No'}")
    print("-------------------------------------------\n")

    # Store request details in state if tool_context provided
    if tool_context:
        # Store individual parameters in state
        tool_context.state["role"] = role
        tool_context.state["industry"] = industry
        tool_context.state["question"] = question
        tool_context.state["resume"] = resume
        tool_context.state["job_description"] = job_description

        # Store structured request details
        tool_context.state["request_details"] = {
            "role": role,
            "industry": industry,
            "question": question,
            "resume": resume,
            "job_description": job_description
        }

        # Initialize final status (will be updated later)
        tool_context.state["final_status"] = "IN_PROGRESS"

    return {
        "role": role,
        "industry": industry,
        "question": question,
        "resume": resume,
        "job_description": job_description
    }


def rate_star_answer(answer: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to evaluate a STAR format answer and provide a numerical rating.

    Args:
        answer: The STAR answer to evaluate
        tool_context: Context for accessing and updating session state

    Returns:
        Dict[str, Any]: Dictionary containing rating and feedback information
    """
    print("\n----------- RATING STAR ANSWER -----------")
    print(f"Answer length: {len(answer)} characters")
    print("-----------------------------------------\n")
    
    tool_context.state["evaluation_performed"] = True
    
    return {
        "message": "Answer has been reviewed. Please provide your rating and feedback."
    }


def finalize_agent_output(rating: float, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Create the final agent output package with all answers and critiques.
    Uses EventActions for reliable state updates.

    This has been refactored from the previous exit_refinement_loop function
    to work with the custom STAROrchestrator, which now handles loop control.

    Args:
        rating: The numerical rating (1.0-5.0) of the STAR answer
        tool_context: Context for tool execution

    Returns:
        Dict containing status information
    """
    print("\n----------- FINALIZING OUTPUT -----------")
    print(f"Rating: {rating}")
    print("------------------------------------------\n")

    # Determine the final status
    if rating >= 4.6:
        final_status = "COMPLETED_HIGH_RATING"
    else:
        final_status = "COMPLETED_MAX_ITERATIONS"

    # Create state updates in one state_delta to ensure atomic updates
    state_delta = {}

    # Store status and final rating
    state_delta["final_status"] = final_status
    state_delta["final_rating"] = rating

    # Ensure we have request details saved
    if "request_details" not in tool_context.state:
        # Extract request details from existing state values
        state_delta["request_details"] = {
            "role": tool_context.state.get("role", ""),
            "industry": tool_context.state.get("industry", ""),
            "question": tool_context.state.get("question", ""),
            "resume": tool_context.state.get("resume", ""),
            "job_description": tool_context.state.get("job_description", "")
        }

    # Get history data from state
    iterations = tool_context.state.get("iterations", [])
    highest_rated_iteration = tool_context.state.get("highest_rated_iteration", 0)

    # Debug log iterations
    print(f"\n[DEBUG finalize_agent_output] Found {len(iterations)} iterations in state")
    for i, iter_item in enumerate(iterations):
        print(f"  Iteration {i+1}: id={iter_item.get('iteration')}, has_answer={iter_item.get('answer') is not None}, has_critique={iter_item.get('critique') is not None}")

    # Get the final answer
    final_answer_raw = tool_context.state.get("current_answer")

    # Parse the final answer into a STARResponse object
    final_star_answer_obj = None
    if isinstance(final_answer_raw, str):
        cleaned_final_answer_raw = _clean_json_string(final_answer_raw)

        try:
            final_star_answer_obj = STARResponse.model_validate_json(cleaned_final_answer_raw)
        except Exception as e:
            print(f"Error parsing final answer as STARResponse: {e}. Using fallback.")
            final_star_answer_obj = STARResponse(
                situation=f"Raw unparsed answer: {final_answer_raw}",
                task="N/A", action="N/A", result="N/A"
            )
    elif isinstance(final_answer_raw, dict):
        try:
            final_star_answer_obj = STARResponse(**final_answer_raw)
        except Exception:
            final_star_answer_obj = STARResponse(
                situation=f"Raw unparsed dict: {str(final_answer_raw)}",
                task="N/A", action="N/A", result="N/A"
            )
    elif isinstance(final_answer_raw, STARResponse):
        final_star_answer_obj = final_answer_raw
    else:
        final_star_answer_obj = STARResponse(
            situation="Placeholder for unexpected answer type",
            task="N/A", action="N/A", result="N/A"
        )

    # Process the iterations for the final output
    iterations_list = iterations.copy()  # Work with a copy to avoid modifying originals

    if not iterations_list and final_star_answer_obj:
        # If no iterations exist, create one with the final answer
        print("No iterations found, creating a placeholder iteration")

        # Get critique feedback from state to use for the rating
        critique_feedback = tool_context.state.get("critique_feedback")
        complete_critique = None

        if critique_feedback:
            # Try to use the actual critique data if available
            if isinstance(critique_feedback, dict) and "rating" in critique_feedback:
                complete_critique = critique_feedback
            elif hasattr(critique_feedback, 'model_dump'):
                complete_critique = critique_feedback.model_dump()

        # If no usable critique found, create a placeholder
        if not complete_critique:
            complete_critique = {
                "rating": rating,
                "structure_feedback": "The answer follows the STAR format with well-defined components.",
                "relevance_feedback": "The answer is relevant to the role and question asked.",
                "specificity_feedback": "The answer includes specific details and examples.",
                "professional_impact_feedback": "The tone is professional and showcases skills appropriately.",
                "suggestions": ["This is a high quality answer that meets the requirements."],
                "raw_critique_text": f"Rating: {rating}"
            }

        # Create an iteration with the final answer and current timestamp
        iterations_list = [{
            "iteration": 1,
            "answer": final_star_answer_obj.model_dump() if hasattr(final_star_answer_obj, 'model_dump') else final_star_answer_obj,
            "critique": complete_critique,
            "rating": rating,
            "timestamp": datetime.datetime.now().isoformat()
        }]
        highest_rated_iteration = 1

        # Update state with the new iterations list
        state_delta["iterations"] = iterations_list
        state_delta["highest_rated_iteration"] = highest_rated_iteration
    else:
        # Process and ensure all existing iterations have complete data
        print(f"Processing {len(iterations_list)} existing iterations")
        iterations_updated = False

        for i, iteration in enumerate(iterations_list):
            # Ensure answer is properly formatted
            answer = iteration.get("answer")
            if answer and not isinstance(answer, dict) and hasattr(answer, 'model_dump'):
                iteration["answer"] = answer.model_dump()
                iterations_updated = True

            # Ensure critique has all required fields
            critique = iteration.get("critique")
            if critique:
                critique_updated = False
                if isinstance(critique, dict) and "rating" in critique:
                    # Fill in any missing required fields with meaningful content
                    if "structure_feedback" not in critique or not critique["structure_feedback"]:
                        critique["structure_feedback"] = "The answer follows the STAR format appropriately."
                        critique_updated = True

                    if "relevance_feedback" not in critique or not critique["relevance_feedback"]:
                        critique["relevance_feedback"] = "The answer is relevant to the role and industry."
                        critique_updated = True

                    if "specificity_feedback" not in critique or not critique["specificity_feedback"]:
                        critique["specificity_feedback"] = "The answer includes specific details and examples."
                        critique_updated = True

                    if "professional_impact_feedback" not in critique or not critique["professional_impact_feedback"]:
                        critique["professional_impact_feedback"] = "The tone is professional and showcases skills effectively."
                        critique_updated = True

                    if "suggestions" not in critique or not critique["suggestions"]:
                        if critique["rating"] >= 4.5:
                            critique["suggestions"] = ["The answer meets high quality standards."]
                        else:
                            critique["suggestions"] = ["Consider adding more specific metrics and examples."]
                        critique_updated = True

                    # Track updated iterations
                    if critique_updated:
                        iterations_updated = True

                    # Check if we need to set the highest_rated_iteration based on existing data
                    iter_rating = critique.get("rating", 0.0)
                    if iter_rating > tool_context.state.get("highest_rating", 0.0):
                        state_delta["highest_rating"] = iter_rating
                        state_delta["highest_rated_iteration"] = iteration.get("iteration", i+1)
                        highest_rated_iteration = iteration.get("iteration", i+1)

            # Make sure each iteration has a timestamp
            if "timestamp" not in iteration:
                iteration["timestamp"] = datetime.datetime.now().isoformat()
                iterations_updated = True

        # Only update iterations in state_delta if we made changes
        if iterations_updated:
            state_delta["iterations"] = iterations_list

    # Create the final output structure
    try:
        final_output_data = AgentFinalOutput(
            final_star_answer=final_star_answer_obj,
            all_iterations=iterations_list,
            highest_rated_iteration=highest_rated_iteration,
            final_rating=rating,
            answer_history=tool_context.state.get("star_responses_history", []),
            critique_history=tool_context.state.get("critiques_history", [])
        )
    except Exception as e:
        print(f"Error creating AgentFinalOutput: {e}. Using fallback format with minimal data.")
        # Create minimal valid data
        final_output_data = {
            "final_star_answer": final_star_answer_obj.model_dump() if hasattr(final_star_answer_obj, 'model_dump') else {
                "situation": "Fallback situation due to validation error",
                "task": "Fallback task due to validation error",
                "action": "Fallback action due to validation error",
                "result": "Fallback result due to validation error"
            },
            "all_iterations": [],
            "highest_rated_iteration": 0,
            "final_rating": rating,
            "answer_history": tool_context.state.get("star_responses_history", []),
            "critique_history": tool_context.state.get("critiques_history", [])
        }

        # Store directly in state_delta (using dict, not model_dump)
        state_delta["final_agent_output_package"] = final_output_data

        # Apply all state updates atomically with EventActions
        if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
            print("Using EventActions.state_delta for proper state updates")
            tool_context.actions.state_delta = state_delta
        else:
            print("EventActions.state_delta not available, falling back to direct state assignment")
            # Direct assignment as fallback
            for key, value in state_delta.items():
                tool_context.state[key] = value

        return {
            "status": "success",
            "message": f"Output finalized. Fallback output package stored with rating: {rating}.",
            "rating": rating
        }

    # Store the final output package in state_delta
    final_output_data_dict = final_output_data.model_dump()
    state_delta["final_agent_output_package"] = final_output_data_dict

    # Debug print final data that's being stored in state
    print(f"\n[DEBUG finalize_agent_output] Final output data with {len(final_output_data_dict.get('all_iterations', []))} iterations")
    print(f"Final rating: {final_output_data_dict.get('final_rating')}")
    print(f"Highest rated iteration: {final_output_data_dict.get('highest_rated_iteration')}")

    # Apply all state updates atomically with EventActions
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        print("Using EventActions.state_delta for proper state updates")
        tool_context.actions.state_delta = state_delta
    else:
        print("EventActions.state_delta not available, falling back to direct state assignment")
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value

    return {
        "status": "success",
        "message": f"Output finalized. Final output package stored with rating: {rating}.",
        "rating": rating,
        "final_agent_output": final_output_data_dict  # Add the output to the tool response directly
    }


def retrieve_final_output_from_state(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Retrieves the final agent output package from the session state and formats it
    according to the EnhancedAgentFinalOutput schema for front-end consumption.
    Uses EventActions to share the final output with the session.

    Args:
        tool_context: Context for tool execution, providing access to session state.

    Returns:
        A dictionary containing the enhanced final output package optimized for front-end use.
    """
    # Silently retrieve final output

    # Get the original final output package from state (legacy format)
    legacy_output_package = tool_context.state.get("final_agent_output_package")

    # Get request details from state
    request_details = tool_context.state.get("request_details", {})
    if not request_details:
        # If not stored as a structure, construct from individual fields
        request_details = {
            "role": tool_context.state.get("role", ""),
            "industry": tool_context.state.get("industry", ""),
            "question": tool_context.state.get("question", ""),
            "resume": tool_context.state.get("resume", ""),
            "job_description": tool_context.state.get("job_description", "")
        }

    # Get final status and error message
    final_status = tool_context.state.get("final_status", "COMPLETED_MAX_ITERATIONS")
    error_message = tool_context.state.get("error_message")

    # Get and validate iterations
    iterations = tool_context.state.get("iterations", [])
    if not iterations and legacy_output_package and "all_iterations" in legacy_output_package:
        iterations = legacy_output_package.get("all_iterations", [])

    # Silently process iterations

    # Convert iterations to the format required for front-end
    interaction_history = []

    for iteration in iterations:
        # Extract the iteration data
        iteration_id = iteration.get("iteration", 0)

        # Get the STAR response
        answer = iteration.get("answer")

        # Handle different types of answer data
        if answer is None:
            # If answer is None, try to use the main answer from state
            # This makes each iteration show the final answer which is better than nothing
            current_answer = tool_context.state.get("current_answer")
            if current_answer:
                if hasattr(current_answer, 'model_dump'):
                    answer = current_answer.model_dump()
                elif isinstance(current_answer, dict):
                    answer = current_answer
                else:
                    # Try parsing if it's a string
                    try:
                        if isinstance(current_answer, str):
                            cleaned_answer = _clean_json_string(current_answer)
                            answer = json.loads(cleaned_answer)
                    except:
                        # Last resort fallback
                        answer = {
                            "situation": str(current_answer)[:200] + "..." if len(str(current_answer)) > 200 else str(current_answer),
                            "task": "Generated from fallback",
                            "action": "Generated from fallback",
                            "result": "Generated from fallback"
                        }
            else:
                # Complete fallback if no answer is available
                answer = {
                    "situation": "No answer data available for this iteration",
                    "task": "Please try again",
                    "action": "The system could not retrieve the answer",
                    "result": "No result was recorded for this iteration"
                }
        elif not isinstance(answer, dict):
            # If it's a Pydantic model, convert to dict
            if hasattr(answer, 'model_dump'):
                answer = answer.model_dump()
            else:
                # Fallback for unexpected types
                answer = {
                    "situation": str(answer)[:200] + "..." if len(str(answer)) > 200 else str(answer),
                    "task": "Generated from non-dict object",
                    "action": "Generated from non-dict object",
                    "result": "Generated from non-dict object"
                }

        # Get the critique
        critique = iteration.get("critique", {})
        if not isinstance(critique, dict):
            # If it's a Pydantic model, convert to dict
            if hasattr(critique, 'model_dump'):
                critique = critique.model_dump()
            else:
                # Fallback for unexpected types
                critique = {
                    "rating": 0.0,
                    "suggestions": ["Error: Could not retrieve suggestions"]
                }

        # Create simplified critique details
        critique_details = {
            "rating": critique.get("rating", 0.0),
            "suggestions": critique.get("suggestions", [])
        }

        # Get or generate timestamp
        timestamp = iteration.get("timestamp")
        if not timestamp:
            timestamp = datetime.datetime.now().isoformat() + "Z"

        # Add to interaction history
        interaction_history.append({
            "iteration_id": iteration_id,
            "star_response": answer,
            "critique_details": critique_details,
            "timestamp_utc": timestamp
        })

    # Sort the interaction history by iteration_id
    interaction_history.sort(key=lambda x: x.get("iteration_id", 0))

    # Get the final STAR answer
    final_star_answer = None
    if legacy_output_package and "final_star_answer" in legacy_output_package:
        final_star_answer = legacy_output_package.get("final_star_answer")
        if hasattr(final_star_answer, 'model_dump'):
            final_star_answer = final_star_answer.model_dump()

    # If no final answer is set, use the last or highest rated answer
    if not final_star_answer:
        # Try to get from highest rated iteration
        highest_rated_iteration = tool_context.state.get("highest_rated_iteration", 0)
        highest_rated_index = next((i for i, it in enumerate(iterations) if it.get("iteration") == highest_rated_iteration), None)

        if highest_rated_index is not None:
            final_star_answer = iterations[highest_rated_index].get("answer", {})
            if hasattr(final_star_answer, 'model_dump'):
                final_star_answer = final_star_answer.model_dump()
        else:
            # Fallback to the last answer
            current_answer = tool_context.state.get("current_answer")
            if current_answer:
                if hasattr(current_answer, 'model_dump'):
                    final_star_answer = current_answer.model_dump()
                else:
                    final_star_answer = current_answer
            else:
                # Last resort fallback
                final_star_answer = {
                    "situation": "No final answer could be retrieved",
                    "task": "Please try again",
                    "action": "The system encountered an error",
                    "result": "No result was generated"
                }

    # Construct the enhanced output using our new format
    try:
        enhanced_output = EnhancedAgentFinalOutput(
            request_details=RequestDetails(**request_details),
            interaction_history=[
                IterationRecord(
                    iteration_id=item["iteration_id"],
                    star_response=STARResponse(**item["star_response"]),
                    critique_details=CritiqueDetails(**item["critique_details"]),
                    timestamp_utc=item["timestamp_utc"]
                )
                for item in interaction_history
            ],
            final_status=final_status,
            error_message=error_message
        )
        enhanced_output_dict = enhanced_output.model_dump()
    except Exception as e:
        # Silently handle error
        # Fallback to direct dictionary construction
        enhanced_output_dict = {
            "request_details": request_details,
            "interaction_history": interaction_history,
            "final_status": final_status,
            "error_message": error_message or "Error constructing enhanced output"
        }

    # Store the enhanced output in state using EventActions
    state_delta = {
        "enhanced_agent_output": enhanced_output_dict,
        "retrieval_timestamp": datetime.datetime.now().isoformat()
    }

    # Apply state updates atomically with EventActions
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        # Use EventActions for atomic updates
        tool_context.actions.state_delta = state_delta
    else:
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value

    # Return the response with our enhanced output
    return {
        "status": "success",
        "message": f"Successfully created enhanced output with {len(interaction_history)} iterations.",
        "retrieved_output": enhanced_output_dict
    }


def initialize_history(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Initialize history structures in session state.
    Uses the EventActions API to properly initialize session state.

    Args:
        tool_context: Context for accessing and updating session state

    Returns:
        Dict with initialization status
    """
    print("\n----------- INITIALIZING HISTORY WITH EVENTS -----------")

    # Create a state delta with our initial values
    state_delta = {
        # Legacy history lists for backward compatibility
        "star_responses_history": [],
        "critiques_history": [],

        # New structured tracking for paired history
        "iterations": [],
        "current_iteration": 0,
        "highest_rated_iteration": 0,
        "highest_rating": 0.0,

        # Initialize status tracking
        "final_status": "IN_PROGRESS",
        "initialization_timestamp": datetime.datetime.now().isoformat()
    }

    # Apply state delta using EventActions
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        print("Using EventActions.state_delta for proper state initialization")
        tool_context.actions.state_delta = state_delta
    else:
        print("EventActions.state_delta not available, falling back to direct state assignment")
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value

    # Double-check that iterations array exists and is empty
    if "iterations" not in tool_context.state or tool_context.state["iterations"] is None:
        print("WARNING: Iterations array not properly initialized. Forcing initialization.")
        tool_context.state["iterations"] = []

    print(f"Initialized state with {len(tool_context.state.get('iterations', []))} iterations")
    print("-------------------------------------------------------\n")

    return {
        "status": "success",
        "message": "Initialized history tracking structures using EventActions"
    }


def append_star_response(tool_context: ToolContext, input_key: str = "current_answer") -> Dict[str, Any]:
    """
    Parse and append STAR response to history.
    Uses EventActions to reliably update session state.

    Args:
        tool_context: Context for accessing and updating session state
        input_key: Key in state containing the STAR response

    Returns:
        Dict with append operation status
    """
    # Record caller information for debugging
    import traceback
    caller_info = traceback.extract_stack()[-2]  # Get the caller frame

    json_string = tool_context.state.get(input_key)
    history_list_key = "star_responses_history"
    parsed_object = None
    raw_input = json_string  # Store original input for debugging

    print(f"\n----------- APPENDING STAR RESPONSE -----------")
    print(f"Called from: {caller_info.name} at {caller_info.filename}:{caller_info.lineno}")

    # Get state of iteration tracking
    iterations = tool_context.state.get('iterations', [])
    current_iter = tool_context.state.get('current_iteration', 0)
    print(f"Current state: iterations={len(iterations)}, current_iteration={current_iter}")
    print(f"Input type: {type(json_string)}")
    print(f"Input length: {len(json_string) if isinstance(json_string, str) else 'N/A'}")
    print(f"Input preview: {json_string[:50]}..." if isinstance(json_string, str) and len(json_string) > 50 else f"Input: {json_string}")

    # IMPORTANT: Create state updates in one state_delta to ensure atomic updates
    state_delta = {}

    # Force the star_responses_history to exist
    if history_list_key not in tool_context.state:
        print(f"Initializing {history_list_key} in state")
        state_delta[history_list_key] = []

    # Force the iterations list to exist
    if "iterations" not in tool_context.state:
        print("Initializing iterations list in state")
        state_delta["iterations"] = []
    else:
        # Copy existing iterations for modification
        state_delta["iterations"] = tool_context.state["iterations"].copy()

    # Parse the STAR response
    if isinstance(json_string, str):
        cleaned_json = _clean_json_string(json_string)

        try:
            parsed_object = STARResponse.model_validate_json(cleaned_json)
            print(f"Successfully parsed STARResponse as Pydantic model")
        except Exception as e:
            print(f"Error parsing STARResponse as Pydantic model: {e}")
            try:
                parsed_object = json.loads(cleaned_json)
                print(f"Successfully parsed STARResponse as JSON dict")
            except Exception as e:
                print(f"Error parsing STARResponse as JSON dict: {e}, using raw string")
                parsed_object = json_string

        # Create or update star_responses_history in state_delta
        if history_list_key in state_delta:
            state_delta[history_list_key].append(parsed_object)
        else:
            legacy_history = tool_context.state.get(history_list_key, []).copy()
            legacy_history.append(parsed_object)
            state_delta[history_list_key] = legacy_history

        # Update the current answer with parsed version
        state_delta[input_key] = parsed_object

        # Determine proper iteration counter based on caller
        if caller_info.name == 'star_generator_with_history':
            # Always set to 1 for initial generator
            current_iteration = 1
        elif caller_info.name == 'star_refiner_with_history':
            # Get highest iteration and increment
            current_iteration = max([it.get("iteration", 0) for it in tool_context.state.get("iterations", [])] + [0]) + 1
        else:
            # Default increment behavior
            current_iteration = tool_context.state.get("current_iteration", 0) + 1

        print(f"Setting iteration counter to {current_iteration} based on caller {caller_info.name}")
        state_delta["current_iteration"] = current_iteration

        # Create a formatted answer object
        answer_obj = None
        if hasattr(parsed_object, 'model_dump'):
            answer_obj = parsed_object.model_dump()
        elif isinstance(parsed_object, dict):
            answer_obj = parsed_object
        else:
            # Create a minimal valid STARResponse if parsing failed
            answer_obj = {
                "situation": str(parsed_object)[:100] + "..." if len(str(parsed_object)) > 100 else str(parsed_object),
                "task": "Parsing error - see raw input",
                "action": "Parsing error - see raw input",
                "result": "Parsing error - see raw input"
            }

        # Store the raw input for reference
        answer_obj["_raw_input"] = raw_input[:200] + "..." if isinstance(raw_input, str) and len(raw_input) > 200 else raw_input

        # Create a new iteration entry with this answer
        iteration_entry = {
            "iteration": current_iteration,
            "answer": answer_obj,
            "critique": None,
            "rating": None,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Process iterations
        iterations_list = state_delta.get("iterations", tool_context.state.get("iterations", [])).copy()

        # Check if this iteration already exists
        existing_index = None
        for i, entry in enumerate(iterations_list):
            if entry.get("iteration") == current_iteration:
                existing_index = i
                break

        if existing_index is not None:
            # Update existing entry
            print(f"Updating existing iteration {current_iteration}")
            iterations_list[existing_index]["answer"] = iteration_entry["answer"]
            iterations_list[existing_index]["timestamp"] = iteration_entry["timestamp"]
        else:
            # Add new entry
            print(f"Adding new iteration {current_iteration}")
            iterations_list.append(iteration_entry)

        # Update iterations in state_delta
        state_delta["iterations"] = iterations_list

        print(f"Current iterations after update: {len(iterations_list)}")

        # Debug print all iterations
        print("Current iterations list:")
        for i, iter_item in enumerate(iterations_list):
            print(f"  Iteration {i+1}: id={iter_item.get('iteration')}, has_answer={iter_item.get('answer') is not None}, has_critique={iter_item.get('critique') is not None}")

    elif json_string is not None: # Input is not a string but exists
        print(f"Non-string input of type {type(json_string)}")
        if history_list_key in state_delta:
            state_delta[history_list_key].append(json_string)
        else:
            legacy_history = tool_context.state.get(history_list_key, []).copy()
            legacy_history.append(json_string)
            state_delta[history_list_key] = legacy_history

        parsed_object = json_string

    # Apply all state updates atomically with EventActions
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        print("Using EventActions.state_delta for proper state updates")
        tool_context.actions.state_delta = state_delta
    else:
        print("EventActions.state_delta not available, falling back to direct state assignment")
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value

    print("---------------------------------------------\n")

    return {
        "status": "success",
        "message": f"Appended STAR response to history for iteration {current_iteration if 'current_iteration' in locals() else tool_context.state.get('current_iteration', 0)}",
        "parsed_type": type(parsed_object).__name__ if parsed_object is not None else "None"
    }


def append_critique(tool_context: ToolContext, input_key: str = "critique_feedback") -> Dict[str, Any]:
    """
    Parse and append critique to history.
    Uses EventActions to reliably update session state.

    Args:
        tool_context: Context for accessing and updating session state
        input_key: Key in state containing the critique

    Returns:
        Dict with append operation status
    """
    print("\n----------- APPENDING CRITIQUE -----------")

    json_string = tool_context.state.get(input_key)
    history_list_key = "critiques_history"
    parsed_object = None
    raw_input = json_string  # Store the original input for debugging

    # Get current iteration for logging
    current_iteration = tool_context.state.get("current_iteration", 0)
    print(f"Current iteration from state: {current_iteration}")

    # IMPORTANT: Create state updates in one state_delta to ensure atomic updates
    state_delta = {}

    # Force history lists to exist in state_delta
    if history_list_key not in tool_context.state:
        print(f"Initializing {history_list_key} in state")
        state_delta[history_list_key] = []

    # Force iterations list to exist in state_delta
    if "iterations" not in tool_context.state:
        print("Initializing iterations list in state")
        state_delta["iterations"] = []
    else:
        # Copy existing iterations for modification
        state_delta["iterations"] = tool_context.state["iterations"].copy()

    # Parse the critique
    if isinstance(json_string, str):
        cleaned_json = _clean_json_string(json_string)

        try:
            parsed_object = Critique.model_validate_json(cleaned_json)
            print(f"Successfully parsed critique as Pydantic model")
        except Exception as e:
            print(f"Error parsing critique as Pydantic model: {e}")
            try:
                parsed_object = json.loads(cleaned_json)
                print(f"Successfully parsed critique as JSON")

                # If parse succeeds but we have a dict without required fields, add them
                if isinstance(parsed_object, dict) and "rating" in parsed_object:
                    # Store the original parsed values before adding defaults
                    original_structure = parsed_object.get("structure_feedback", "")
                    original_relevance = parsed_object.get("relevance_feedback", "")
                    original_specificity = parsed_object.get("specificity_feedback", "")
                    original_impact = parsed_object.get("professional_impact_feedback", "")
                    original_suggestions = parsed_object.get("suggestions", [])

                    # Add required fields that might be missing for validation - retain original content when available
                    if not parsed_object.get("structure_feedback"):
                        parsed_object["structure_feedback"] = original_structure or "Auto-generated for validation"
                    if not parsed_object.get("relevance_feedback"):
                        parsed_object["relevance_feedback"] = original_relevance or "Auto-generated for validation"
                    if not parsed_object.get("specificity_feedback"):
                        parsed_object["specificity_feedback"] = original_specificity or "Auto-generated for validation"
                    if not parsed_object.get("professional_impact_feedback"):
                        parsed_object["professional_impact_feedback"] = original_impact or "Auto-generated for validation"
                    if not parsed_object.get("suggestions"):
                        parsed_object["suggestions"] = original_suggestions or ["Auto-generated for validation"]

                    # Store raw critique text for preservation of original format
                    if "raw_critique_text" not in parsed_object:
                        parsed_object["raw_critique_text"] = raw_input
            except Exception as e2:
                print(f"Error parsing critique as JSON: {e2}, using raw string")
                parsed_object = json_string

        # Update critiques_history in state_delta
        if history_list_key in state_delta:
            state_delta[history_list_key].append(parsed_object)
        else:
            legacy_history = tool_context.state.get(history_list_key, []).copy()
            legacy_history.append(parsed_object)
            state_delta[history_list_key] = legacy_history

        # Update the critique key with parsed version
        state_delta[input_key] = parsed_object

        # Process iterations
        iterations_list = state_delta.get("iterations", tool_context.state.get("iterations", [])).copy()

        # Find the iteration entry for the current iteration
        found = False
        rating = None

        for entry in iterations_list:
            if entry.get("iteration") == current_iteration:
                # Process critique based on its type
                if isinstance(parsed_object, dict) and "rating" in parsed_object:
                    # Use the existing parsed_object - it already has all required fields
                    entry["critique"] = parsed_object
                    rating = parsed_object["rating"]

                    # Add raw critique text if not present
                    if "raw_critique_text" not in parsed_object:
                        entry["critique"]["raw_critique_text"] = raw_input
                elif hasattr(parsed_object, 'model_dump'):
                    # If it's a Pydantic model, convert to dict
                    model_dict = parsed_object.model_dump()

                    # Add raw critique text
                    if "raw_critique_text" not in model_dict or not model_dict["raw_critique_text"]:
                        model_dict["raw_critique_text"] = raw_input

                    entry["critique"] = model_dict
                    rating = getattr(parsed_object, "rating", None)
                else:
                    # Fallback
                    entry["critique"] = {
                        "rating": 0.0,
                        "raw_critique_text": raw_input,
                        "structure_feedback": "Error parsing critique",
                        "relevance_feedback": "Error parsing critique",
                        "specificity_feedback": "Error parsing critique",
                        "professional_impact_feedback": "Error parsing critique",
                        "suggestions": ["Error parsing critique"]
                    }
                    rating = 0.0

                entry["rating"] = rating
                found = True
                print(f"Updated existing iteration {current_iteration} with critique")
                break

        # Update or create an iteration entry if needed
        if not found and current_iteration > 0:
            print(f"No existing iteration {current_iteration} found, creating new entry")

            # Create a new entry with just the critique
            # Determine the critique object to use
            if isinstance(parsed_object, dict) and "rating" in parsed_object:
                # Use the parsed_object directly - it's already complete with required fields
                critique_obj = parsed_object

                # Ensure raw critique text is preserved
                if "raw_critique_text" not in critique_obj:
                    critique_obj["raw_critique_text"] = raw_input

                rating = critique_obj["rating"]
            elif hasattr(parsed_object, 'model_dump'):
                # Convert Pydantic model to dict
                critique_obj = parsed_object.model_dump()

                # Ensure raw critique text is preserved
                if "raw_critique_text" not in critique_obj or not critique_obj["raw_critique_text"]:
                    critique_obj["raw_critique_text"] = raw_input

                rating = getattr(parsed_object, "rating", None)
            else:
                # Fallback for other object types
                critique_obj = {
                    "rating": 0.0,
                    "raw_critique_text": raw_input,
                    "structure_feedback": "Error parsing critique",
                    "relevance_feedback": "Error parsing critique",
                    "specificity_feedback": "Error parsing critique",
                    "professional_impact_feedback": "Error parsing critique",
                    "suggestions": ["Error parsing critique"]
                }
                rating = 0.0

            # Try to find the current answer from state to include
            current_answer = tool_context.state.get("current_answer")
            answer_obj = None

            if current_answer:
                if hasattr(current_answer, 'model_dump'):
                    answer_obj = current_answer.model_dump()
                elif isinstance(current_answer, dict):
                    answer_obj = current_answer
                elif isinstance(current_answer, str):
                    # Try to parse the string as JSON
                    try:
                        cleaned_answer = _clean_json_string(current_answer)
                        answer_obj = json.loads(cleaned_answer)
                    except:
                        # If parsing fails, use the raw string
                        answer_obj = {
                            "situation": current_answer[:200] + "..." if len(current_answer) > 200 else current_answer,
                            "task": "Parsed from raw string",
                            "action": "Parsed from raw string",
                            "result": "Parsed from raw string"
                        }

            # Create the iteration entry with timestamp
            entry = {
                "iteration": current_iteration,
                "answer": answer_obj,  # Include the answer object if available
                "critique": critique_obj,
                "rating": rating,
                "timestamp": datetime.datetime.now().isoformat()
            }

            iterations_list.append(entry)
            if answer_obj:
                print(f"Added new iteration entry with answer and critique for iteration {current_iteration}")
            else:
                print(f"Added new iteration entry with just critique for iteration {current_iteration}")

        # Update iterations in state_delta
        state_delta["iterations"] = iterations_list

        # Update highest rating if this is better
        if rating is not None:
            highest_rating = tool_context.state.get("highest_rating", 0.0)
            if rating > highest_rating:
                state_delta["highest_rating"] = rating
                state_delta["highest_rated_iteration"] = current_iteration
                print(f"Updated highest rating to {rating} from iteration {current_iteration}")

    elif json_string is not None: # Input is not a string but exists
        print(f"Non-string critique input of type {type(json_string)}")

        if history_list_key in state_delta:
            state_delta[history_list_key].append(json_string)
        else:
            legacy_history = tool_context.state.get(history_list_key, []).copy()
            legacy_history.append(json_string)
            state_delta[history_list_key] = legacy_history

        parsed_object = json_string

    # Apply all state updates atomically with EventActions
    if hasattr(tool_context, 'actions') and hasattr(tool_context.actions, 'state_delta'):
        print("Using EventActions.state_delta for proper state updates")
        tool_context.actions.state_delta = state_delta

        # Debug iterations after update
        print(f"Current iterations count: {len(state_delta.get('iterations', []))}")
        for i, iter_item in enumerate(state_delta.get("iterations", [])):
            print(f"  Iteration {i+1}: id={iter_item.get('iteration')}, "
                  f"has_answer={iter_item.get('answer') is not None}, "
                  f"has_critique={iter_item.get('critique') is not None}")
    else:
        print("EventActions.state_delta not available, falling back to direct state assignment")
        # Direct assignment as fallback
        for key, value in state_delta.items():
            tool_context.state[key] = value

    print("---------------------------------------------\n")

    return {
        "status": "success",
        "message": f"Appended critique to history for iteration {current_iteration}",
        "parsed_type": type(parsed_object).__name__ if parsed_object is not None else "None"
    }


def refine_star_answer(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool that refines a STAR answer based on critique feedback.
    This simply delegates to the refiner agent with the current answer.

    Args:
        tool_context: Context for accessing state and delegating to agents

    Returns:
        Dict containing the refined answer
    """
    print("\n----------- REFINING STAR ANSWER -----------")
    current_answer = tool_context.state.get("current_answer")

    # Import the refiner agent
    try:
        # Get the parent directory of this module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        # Load refiner module from relative path
        refiner_path = os.path.join(parent_dir, "sample_agent", "subagents", "refiner", "agent.py")
        spec = importlib.util.spec_from_file_location("refiner_agent", refiner_path)
        refiner_module = importlib.util.module_from_spec(spec)
        sys.modules["refiner_agent"] = refiner_module
        spec.loader.exec_module(refiner_module)

        # Get the refiner agent from the module
        refiner_agent = refiner_module.star_refiner
    except Exception as e:
        print(f"Error importing refiner agent: {e}")
        return {
            "status": "error",
            "message": f"Failed to import refiner agent: {e}"
        }

    # Delegate to the refiner agent
    try:
        refined_answer = refiner_agent.invoke(current_answer)

        # Store in state
        if refined_answer:
            # Process the refined answer
            tool_context.state["current_answer"] = refined_answer

            # Add to history using append_star_response tool
            append_star_response(tool_context)

            print(f"Successfully refined the STAR answer")
        else:
            print(f"Refiner returned empty response")

    except Exception as e:
        print(f"Error during refinement: {e}")
        return {
            "status": "error",
            "message": f"Refinement failed: {e}"
        }

    print("-----------------------------------------\n")

    return {
        "status": "success",
        "message": "Successfully refined the STAR answer",
        "refined_answer": tool_context.state["current_answer"]
    }


def _clean_json_string(json_string: str) -> str:
    """
    Helper function to clean JSON strings from markdown formatting.

    Args:
        json_string: The JSON string to clean

    Returns:
        The cleaned JSON string
    """
    cleaned_json = json_string
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[len("```json"):].strip()
    elif cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[len("```"):].strip()
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-len("```")].strip()

    return cleaned_json