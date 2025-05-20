"""
STAR Answer Orchestrator Agent

This module defines a custom orchestrator agent that manages the STAR answer generation workflow
with precise control over the refinement process.
"""

import logging
import json
import traceback
import datetime
from typing import AsyncGenerator, Optional
from typing_extensions import override

from google.adk.agents import Agent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from .timing import TimingTracker, time_operation
from .tools import retrieve_final_output_from_state
from .parsing_utils import parse_llm_json_output, parse_critique_feedback, parse_star_answer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_iteration_info(ctx, iteration_number):
    """Update the current iteration info in the state."""
    if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
        state_delta = {
            "current_iteration": iteration_number + 1  # Prepare for next iteration
        }
        ctx.actions.state_delta = state_delta
    else:
        # Direct state updates as fallback
        ctx.session.state["current_iteration"] = iteration_number + 1

class STAROrchestrator(BaseAgent):
    """
    Custom agent for STAR answer generation with conditional refinement.
    
    This agent orchestrates the workflow of generating, critiquing, and
    optionally refining STAR format answers based on rating thresholds.
    
    Unlike the standard LoopAgent, this custom agent allows skipping the refiner
    when a high rating is achieved, preventing unnecessary processing.
    """
    
    # Field declarations for Pydantic
    input_collector: Agent
    star_generator: Agent
    star_critique: Agent
    star_refiner: Agent

    # Configuration
    rating_threshold: float
    max_iterations: int
    timing_tracker: TimingTracker

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        input_collector: Agent,
        star_generator: Agent,
        star_critique: Agent,
        star_refiner: Agent,
        rating_threshold: float = 4.6,
        max_iterations: int = 3,
    ):
        """
        Initialize the STAR Orchestrator agent.

        Args:
            name: Name of the agent
            input_collector: Agent to collect user inputs
            star_generator: Agent to generate initial STAR answer
            star_critique: Agent to critique STAR answers
            star_refiner: Agent to refine STAR answers
            rating_threshold: Rating threshold to skip refinement (default: 4.6)
            max_iterations: Maximum refinement iterations (default: 3)
        """
        # Store all sub-agents
        super().__init__(
            name=name,
            input_collector=input_collector,
            star_generator=star_generator,
            star_critique=star_critique,
            star_refiner=star_refiner,
            rating_threshold=rating_threshold,
            max_iterations=max_iterations,
            timing_tracker=TimingTracker(),
            sub_agents=[
                input_collector,
                star_generator,
                star_critique,
                star_refiner
            ],
            description="Custom orchestrator for STAR format answer generation with conditional refinement",
        )
    
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom STAR workflow with conditional refinement.

        This method provides precise control over the execution flow, allowing us to
        skip the refiner when a high rating is achieved.

        Args:
            ctx: The invocation context containing state and other runtime information

        Yields:
            Events from the sub-agents as they are generated
        """
        print("========= ORCHESTRATOR IS RUNNING =========")
        print(f"Rating threshold is: {self.rating_threshold}")
        print(f"Max iterations is: {self.max_iterations}")
        logger.info(f"[{self.name}] Starting STAR answer generation workflow.")
        self.timing_tracker.reset()  # Reset timing for new request
        self.timing_tracker.start("total_workflow")

        # Step 1: Direct initialization - No agent needed
        logger.info(f"[{self.name}] Directly initializing history state...")
        # Initialize state directly
        history_state = {
            "iterations": [],  # Legacy (kept for backward compatibility)
            "full_iteration_history": [],  # Main tracking structure
            "current_iteration": 0,  # Will be set to 1 before first STAR generation
            "highest_rated_iteration": 0,
            "highest_rating": 0.0,
            "final_status": "IN_PROGRESS"
        }

        # Apply state update
        if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
            ctx.actions.state_delta = history_state
            # Also update the session state for immediate use
            for key, value in history_state.items():
                ctx.session.state[key] = value
        else:
            # Direct state update
            for key, value in history_state.items():
                ctx.session.state[key] = value

        logger.info(f"[{self.name}] History state initialized directly")

        # Step 2: Collect inputs
        logger.info(f"[{self.name}] Collecting inputs...")
        with time_operation(self.timing_tracker, "input_collector"):
            async for event in self.input_collector.run_async(ctx):
                # Log event information for debugging
                logger.info(f"[{self.name}] Event from input_collector: {event.author} (has_content={event.content is not None})")
                # Forward events from the input_collector
                yield event
        
        # Check if we have the required inputs before proceeding
        if not ctx.session.state.get("role") or not ctx.session.state.get("industry") or not ctx.session.state.get("question"):
            logger.error(f"[{self.name}] Missing required inputs. Aborting workflow.")

            # Directly prepare and yield final error output
            error_payload = self.prepare_final_json_for_ui(
                full_history=ctx.session.state.get("full_iteration_history", []),
                final_status=ctx.session.state.get("final_status", "ERROR_INPUT_VALIDATION"),
                final_answer=None, # No successful answer
                final_rating=0.0,
                highest_rated_iteration_num=ctx.session.state.get("highest_rated_iteration", 0),
                timing_data=self.timing_tracker.get_all_timings(),
                error_message=ctx.session.state.get("error_message")
            )
            logger.info(f"[{self.name}] Yielding final error output directly.")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=error_payload)]),
                is_final_response=True
            )
            return
        
        # Step 3: Generate initial STAR answer
        logger.info(f"[{self.name}] Generating initial STAR answer...")

        # Set the initial iteration to 1 for the first STAR answer
        if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
            ctx.actions.state_delta = {"current_iteration": 1}
        else:
            ctx.session.state["current_iteration"] = 1

        try:
            with time_operation(self.timing_tracker, "star_generator"):
                async for event in self.star_generator.run_async(ctx):
                    # Log event information for debugging
                    logger.info(f"[{self.name}] Event from star_generator: {event.author} (has_content={event.content is not None})")
                    # Forward events from the star_generator
                    yield event
        except Exception as e:
            logger.error(f"[{self.name}] Star generator failed: {e}")
            # Directly prepare and yield final error output
            error_payload = self.prepare_final_json_for_ui(
                full_history=ctx.session.state.get("full_iteration_history", []),
                final_status=ctx.session.state.get("final_status", "ERROR_AGENT_PROCESSING"),
                final_answer=None, # No successful answer
                final_rating=0.0,
                highest_rated_iteration_num=ctx.session.state.get("highest_rated_iteration", 0),
                timing_data=self.timing_tracker.get_all_timings(),
                error_message=ctx.session.state.get("error_message")
            )
            logger.info(f"[{self.name}] Yielding final error output directly after agent failure.")
            yield Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=types.Content(parts=[types.Part(text=error_payload)]),
                is_final_response=True
            )
            return

        # Step 4: Iterative refinement loop with conditional execution
        iteration = 1
        final_rating = 0.0
        
        while iteration <= self.max_iterations:
            logger.info(f"[{self.name}] Starting iteration {iteration} (rating threshold: {self.rating_threshold})")

            # Run critique
            logger.info(f"[{self.name}] Running critique for iteration {iteration}...")
            try:
                with time_operation(self.timing_tracker, f"star_critique_iteration_{iteration}"):
                    async for event in self.star_critique.run_async(ctx):
                        # Log event information for debugging
                        logger.info(f"[{self.name}] Event from star_critique: {event.author} (has_content={event.content is not None})")
                        # Forward events from the star_critique
                        yield event
            except Exception as e:
                logger.error(f"[{self.name}] Star critique failed: {e}")
                # Directly prepare and yield final error output
                error_payload = self.prepare_final_json_for_ui(
                    full_history=ctx.session.state.get("full_iteration_history", []),
                    final_status=ctx.session.state.get("final_status", "ERROR_AGENT_PROCESSING"),
                    final_answer=None, # No successful answer
                    final_rating=0.0,
                    highest_rated_iteration_num=ctx.session.state.get("highest_rated_iteration", 0),
                    timing_data=self.timing_tracker.get_all_timings(),
                    error_message=ctx.session.state.get("error_message")
                )
                logger.info(f"[{self.name}] Yielding final error output directly after agent failure.")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=error_payload)]),
                    is_final_response=True
                )
                return
            
            # Get the latest state after critique agent has finished
            # The state should be updated via state_delta by the append_critique tool
            current_session = ctx.session
            logger.info(f"[{self.name}] Post-critique state keys: {list(current_session.state.keys())}")

            # Debug: Check critique feedback directly
            critique_feedback_raw = current_session.state.get("critique_feedback", {})
            print(f"[ORCHESTRATOR DEBUG] Critique feedback type: {type(critique_feedback_raw)}")

            # Parse critique feedback using our centralized utility
            try:
                parsed_debug_critique = parse_critique_feedback(critique_feedback_raw)
                print(f"[ORCHESTRATOR DEBUG] Parsed critique feedback: {parsed_debug_critique}")

                if parsed_debug_critique:
                    direct_rating = parsed_debug_critique.get('rating', 'NOT FOUND')
                    print(f"[ORCHESTRATOR DEBUG] Direct rating from critique_feedback: {direct_rating}")
            except Exception as e:
                print(f"[ORCHESTRATOR DEBUG] Failed to parse critique feedback with utility: {e}")
                # Continue with execution - this is just for debugging

            # Get the rating from the current iteration
            rating = 0.0
            iterations = current_session.state.get("iterations", [])
            logger.info(f"[{self.name}] Found {len(iterations)} iterations in state")

            # Debug: Print the structure of the iterations
            print(f"[ORCHESTRATOR DEBUG] All iterations: {iterations}")
            if iterations:
                print(f"[ORCHESTRATOR DEBUG] First iteration structure: {iterations[0]}")
                print(f"[ORCHESTRATOR DEBUG] First iteration keys: {list(iterations[0].keys()) if isinstance(iterations[0], dict) else 'Not a dict'}")

            # Debug current_iteration in state vs loop variable
            state_current_iter = current_session.state.get("current_iteration", "NOT FOUND")
            print(f"[ORCHESTRATOR DEBUG] State current_iteration: {state_current_iter}")
            print(f"[ORCHESTRATOR DEBUG] Loop iteration variable: {iteration}")

            # Process the critique feedback from the state (set by star_critique agent)
            critique_feedback_raw = ctx.session.state.get("critique_feedback")

            # Use centralized parsing utility for critique feedback
            logger.info(f"[{self.name}] Parsing critique feedback using centralized utility")

            # Parse the critique feedback using our utility function
            parsed_critique = parse_critique_feedback(critique_feedback_raw)

            # Extract the rating and use the parsed critique for history
            rating = parsed_critique.get("rating", 0.0)
            critique_details_for_history = parsed_critique

            logger.info(f"[{self.name}] Successfully parsed critique feedback. Rating: {rating}")

            final_rating = rating

            # --- Start: Retrieve and parse the raw answer string from state ---
            raw_answer_string_key = self.star_generator.output_key if iteration == 1 else self.star_refiner.output_key
            raw_answer_string = ctx.session.state.get(raw_answer_string_key)

            # Use centralized parsing utility for STAR answer
            logger.info(f"[{self.name}] Iteration {iteration}: Parsing STAR answer using centralized utility from key '{raw_answer_string_key}'")

            # Parse the STAR answer using our utility function
            parsed_answer_obj = parse_star_answer(raw_answer_string)

            logger.info(f"[{self.name}] Successfully parsed STAR answer with keys: {list(parsed_answer_obj.keys()) if isinstance(parsed_answer_obj, dict) else 'Not a dict'}")
            # --- End: Retrieve and parse the raw answer string ---

            # --- Start: Define iteration_entry and append to full_iteration_history ---
            iteration_entry = {
                "iteration_number": iteration,
                "answer": parsed_answer_obj,                   # Use the newly parsed answer object
                "critique": critique_details_for_history,      # Use the critique details parsed earlier
                "rating": rating,                              # Assumed to be defined from critique processing
                "timestamp": datetime.datetime.now().isoformat(),
            }

            current_history_list = ctx.session.state.get("full_iteration_history", [])
            if not isinstance(current_history_list, list):
                logger.warning(f"[{self.name}] Iteration {iteration}: 'full_iteration_history' in state was not a list. Re-initializing to empty list for history construction.")
                current_history_list = []

            # Initialize new_history_list safely as a copy of current_history_list
            new_history_list = list(current_history_list) 
            try:
                # Attempt to append the current iteration's entry
                new_history_list.append(iteration_entry)
            except Exception as e:
                logger.error(f"[{self.name}] Iteration {iteration}: Failed to append iteration_entry to history. Error: {e}. This iteration's data might be lost from history.")
                # new_history_list remains as it was before the failed append (i.e., history up to the previous iteration)
                # Depending on requirements, one might choose to re-raise or handle more explicitly.

            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"full_iteration_history": new_history_list}
                ctx.session.state["full_iteration_history"] = new_history_list # Ensure current context sees it too
            else:
                ctx.session.state["full_iteration_history"] = new_history_list

            # Debug log for the appended item
            logger.info(f"[{self.name}] Added iteration {iteration} details to full_iteration_history.")
            if new_history_list:
                last_entry = new_history_list[-1]
                print(f"[ORCHESTRATOR DEBUG] Last item in full_iteration_history: iteration_number={last_entry.get('iteration_number')}, rating={last_entry.get('rating')}, answer_keys_present={list(last_entry.get('answer').keys()) if isinstance(last_entry.get('answer'), dict) else type(last_entry.get('answer'))}, critique_keys_present={list(last_entry.get('critique').keys()) if isinstance(last_entry.get('critique'), dict) else type(last_entry.get('critique'))}")
            else:
                print("[ORCHESTRATOR DEBUG] full_iteration_history is empty after trying to append.")
            # --- End: Define iteration_entry and append to full_iteration_history ---

            # Update final_rating with the latest one, to be used for threshold check
            # highest_rating can be updated here if needed, or keep original logic if it's managed elsewhere
            highest_rating = max(ctx.session.state.get("highest_rating", 0.0), final_rating)
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"highest_rating": highest_rating} # Persist if changed
            else:
                ctx.session.state["highest_rating"] = highest_rating

            threshold_check_rating = final_rating # Use the most recent rating for the decision
            logger.info(f"[{self.name}] Current rating: {final_rating}, Highest rating so far: {highest_rating}")
            logger.info(f"[{self.name}] Using rating {threshold_check_rating} for threshold check (threshold: {self.rating_threshold})")

            # Check if rating meets threshold to skip refinement
            print(f"[ORCHESTRATOR DEBUG] Checking rating {threshold_check_rating} >= {self.rating_threshold}")
            if threshold_check_rating >= self.rating_threshold:
                logger.info(f"[{self.name}] Rating {threshold_check_rating} meets threshold {self.rating_threshold}. Stopping refinement.")
                print(f"[ORCHESTRATOR DEBUG] Rating meets threshold! Breaking refinement loop")

                # Use state_delta for atomic updates if available
                if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                    state_delta = {
                        "final_status": "COMPLETED_HIGH_RATING",
                        "final_rating": rating,
                        "current_iteration": iteration  # Don't increment, we're done
                    }
                    ctx.actions.state_delta = state_delta
                else:
                    # Direct state updates as fallback
                    ctx.session.state["final_status"] = "COMPLETED_HIGH_RATING"
                    ctx.session.state["final_rating"] = rating
                    ctx.session.state["current_iteration"] = iteration  # Don't increment

                # Break the loop to skip refinement
                break
            
            # Rating is below threshold, run refiner
            logger.info(f"[{self.name}] Rating {threshold_check_rating} is below threshold {self.rating_threshold}. Running refiner...")

            # Increment iteration for the NEXT STAR answer before running refiner
            iteration += 1

            # Update state with new iteration number for the next answer
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"current_iteration": iteration}
            else:
                ctx.session.state["current_iteration"] = iteration

            try:
                with time_operation(self.timing_tracker, f"star_refiner_iteration_{iteration-1}"):
                    async for event in self.star_refiner.run_async(ctx):
                        # Log event information for debugging
                        logger.info(f"[{self.name}] Event from star_refiner: {event.author} (has_content={event.content is not None})")
                        # Forward events from the star_refiner
                        yield event
            except Exception as e:
                logger.error(f"[{self.name}] Star refiner failed: {e}")
                # Directly prepare and yield final error output
                error_payload = self.prepare_final_json_for_ui(
                    full_history=ctx.session.state.get("full_iteration_history", []),
                    final_status=ctx.session.state.get("final_status", "ERROR_AGENT_PROCESSING"),
                    final_answer=None, # No successful answer
                    final_rating=0.0,
                    highest_rated_iteration_num=ctx.session.state.get("highest_rated_iteration", 0),
                    timing_data=self.timing_tracker.get_all_timings(),
                    error_message=ctx.session.state.get("error_message")
                )
                logger.info(f"[{self.name}] Yielding final error output directly after agent failure.")
                yield Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=types.Content(parts=[types.Part(text=error_payload)]),
                    is_final_response=True
                )
                return
        
        # Check if we finished due to max iterations
        if iteration > self.max_iterations:
            logger.info(f"[{self.name}] Reached max iterations ({self.max_iterations}). Completing workflow.")

            # Use state_delta for atomic updates if available
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                state_delta = {
                    "final_status": "COMPLETED_MAX_ITERATIONS",
                    "final_rating": final_rating
                }
                ctx.actions.state_delta = state_delta
            else:
                # Direct state updates as fallback
                ctx.session.state["final_status"] = "COMPLETED_MAX_ITERATIONS"
                ctx.session.state["final_rating"] = final_rating
        
        # Step 5: Complete workflow timing and add to state BEFORE output retriever
        # Since we need timing data in the output retriever, we'll use direct state update
        workflow_timing = self.timing_tracker.end("total_workflow")
        timing_data = self.timing_tracker.get_timings()
        logger.info(f"[{self.name}] Collected timing data before output retriever: {timing_data}")
        print(f"[ORCHESTRATOR DEBUG] Timing data collected: {timing_data}")

        # Add timing data directly to state to make it immediately available
        ctx.session.state["timing_data"] = timing_data
        logger.info(f"[{self.name}] Added timing data directly to state before output retriever")
        print(f"[ORCHESTRATOR DEBUG] Added timing_data to state. State type: {type(ctx.session.state)}")

        # NEW: Prepare the final JSON payload using our Python function
        logger.info(f"[{self.name}] Calling Python function to prepare final JSON payload for UI...")
        final_json_string_for_ui = retrieve_final_output_from_state(ctx) # tool_context is ctx here

        print(f"[ORCHESTRATOR PRE-LOG DEBUG] Type of final_json_string_for_ui: {type(final_json_string_for_ui)}, Len: {len(final_json_string_for_ui) if isinstance(final_json_string_for_ui, str) else 'N/A'}")
        logger.info(f"[{self.name}] Orchestrator received JSON string from tool (len: {len(final_json_string_for_ui)}). Snippet: {final_json_string_for_ui[:1000]}...")
        
        # Yield the final JSON payload directly
        logger.info(f"[{self.name}] Orchestrator yielding final JSON payload directly (len: {len(final_json_string_for_ui)}). Snippet: {final_json_string_for_ui[:500]}...")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text=final_json_string_for_ui)])
        )

        logger.info(f"[{self.name}] STAR Orchestrator finished.")


def update_iteration_info(ctx: InvocationContext, current_iteration: int) -> None:
    """
    Helper function to update iteration information in state.
    Uses EventActions.state_delta when available for atomic updates.

    Args:
        ctx: Invocation context with access to session state
        current_iteration: The current iteration number
    """
    # Check if state_delta is available via actions
    if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
        # Use state_delta for atomic update
        ctx.actions.state_delta = {
            "current_iteration": current_iteration,
            # Additional state updates can be added here if needed
        }
    else:
        # Direct state update as fallback
        ctx.session.state["current_iteration"] = current_iteration