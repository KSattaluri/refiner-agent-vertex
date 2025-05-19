"""
STAR Answer Orchestrator Agent

This module defines a custom orchestrator agent that manages the STAR answer generation workflow
with precise control over the refinement process.
"""

import logging
import json
import traceback
from typing import AsyncGenerator, Optional
from typing_extensions import override

from google.adk.agents import Agent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from .timing import TimingTracker, time_operation

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
    initialize_agent: Agent
    input_collector: Agent
    star_generator: Agent
    star_critique: Agent
    star_refiner: Agent
    output_retriever: Agent
    
    # Configuration
    rating_threshold: float
    max_iterations: int
    timing_tracker: TimingTracker

    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(
        self,
        name: str,
        initialize_agent: Agent,
        input_collector: Agent,
        star_generator: Agent,
        star_critique: Agent,
        star_refiner: Agent,
        output_retriever: Agent,
        rating_threshold: float = 4.6,
        max_iterations: int = 3,
    ):
        """
        Initialize the STAR Orchestrator agent.
        
        Args:
            name: Name of the agent
            initialize_agent: Agent to initialize history
            input_collector: Agent to collect user inputs
            star_generator: Agent to generate initial STAR answer
            star_critique: Agent to critique STAR answers
            star_refiner: Agent to refine STAR answers
            output_retriever: Agent to retrieve and format final output
            rating_threshold: Rating threshold to skip refinement (default: 4.6)
            max_iterations: Maximum refinement iterations (default: 3)
        """
        # Store all sub-agents
        super().__init__(
            name=name,
            initialize_agent=initialize_agent,
            input_collector=input_collector,
            star_generator=star_generator,
            star_critique=star_critique,
            star_refiner=star_refiner,
            output_retriever=output_retriever,
            rating_threshold=rating_threshold,
            max_iterations=max_iterations,
            timing_tracker=TimingTracker(),
            sub_agents=[
                initialize_agent,
                input_collector,
                star_generator,
                star_critique,
                star_refiner,
                output_retriever
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

        # Initialize full_iteration_history in state if it doesn't exist
        if "full_iteration_history" not in ctx.session.state:
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                # Initialize via state_delta if possible, though direct init is often fine here
                # as it's at the start of the orchestrator's run for this request.
                # However, initialize_agent might be a more canonical place.
                # For this change, we ensure it's present before use.
                ctx.actions.state_delta = {"full_iteration_history": []}
                # Ensure it's reflected in current session state for immediate use if state_delta is deferred
                if "full_iteration_history" not in ctx.session.state: 
                     ctx.session.state["full_iteration_history"] = []
            else:
                ctx.session.state["full_iteration_history"] = []
        elif isinstance(ctx.session.state.get("full_iteration_history"), list):
            # If it exists and is a list, clear it for this new workflow run.
            # This assumes one orchestrator instance handles multiple requests sequentially
            # and state might persist if not cleared by initialize_agent properly.
            # A safer approach is for initialize_agent to always set/reset this.
            # For now, let's ensure it's empty if it was already a list.
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"full_iteration_history": []}
                ctx.session.state["full_iteration_history"] = [] # Reflect for current execution
            else:
                ctx.session.state["full_iteration_history"] = []
        else:
            # If it exists but is not a list, force it to be an empty list
            logger.warning(f"[{self.name}] 'full_iteration_history' in state was not a list. Re-initializing.")
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"full_iteration_history": []}
                ctx.session.state["full_iteration_history"] = [] # Reflect for current execution
            else:
                ctx.session.state["full_iteration_history"] = []

        # Step 1: Initialize history
        logger.info(f"[{self.name}] Initializing history...")
        with time_operation(self.timing_tracker, "initialize_agent"):
            async for event in self.initialize_agent.run_async(ctx):
                # Log event information for debugging
                logger.info(f"[{self.name}] Event from initialize_agent: {event.author} (has_content={event.content is not None})")
                # Forward events from the initialize_agent
                yield event

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

            # Update state with error status using state_delta when available
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                state_delta = {
                    "final_status": "ERROR_INPUT_VALIDATION",
                    "error_message": "Missing required inputs: role, industry, or question"
                }
                ctx.actions.state_delta = state_delta
            else:
                # Direct state updates as fallback
                ctx.session.state["final_status"] = "ERROR_INPUT_VALIDATION"
                ctx.session.state["error_message"] = "Missing required inputs: role, industry, or question"

            # Run output_retriever to get formatted error information
            logger.info(f"[{self.name}] Retrieving error output...")
            async for event in self.output_retriever.run_async(ctx):
                yield event

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
            # Update state with error status
            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                state_delta = {
                    "final_status": "ERROR_AGENT_PROCESSING",
                    "error_message": f"Star generator failed: {str(e)}"
                }
                ctx.actions.state_delta = state_delta
            else:
                ctx.session.state["final_status"] = "ERROR_AGENT_PROCESSING"
                ctx.session.state["error_message"] = f"Star generator failed: {str(e)}"

            # Run output retriever to format error
            async for event in self.output_retriever.run_async(ctx):
                yield event
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
                # Update state with error status
                if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                    state_delta = {
                        "final_status": "ERROR_AGENT_PROCESSING",
                        "error_message": f"Star critique failed: {str(e)}"
                    }
                    ctx.actions.state_delta = state_delta
                else:
                    ctx.session.state["final_status"] = "ERROR_AGENT_PROCESSING"
                    ctx.session.state["error_message"] = f"Star critique failed: {str(e)}"

                # Run output retriever to format error
                async for event in self.output_retriever.run_async(ctx):
                    yield event
                return
            
            # Get the latest state after critique agent has finished
            # The state should be updated via state_delta by the append_critique tool
            current_session = ctx.session
            logger.info(f"[{self.name}] Post-critique state keys: {list(current_session.state.keys())}")

            # Debug: Check critique feedback directly
            critique_feedback = current_session.state.get("critique_feedback", {})
            print(f"[ORCHESTRATOR DEBUG] Critique feedback type: {type(critique_feedback)}")

            # Parse critique feedback if it's a string
            if isinstance(critique_feedback, str):
                try:
                    # Clean markdown blocks if present
                    cleaned = critique_feedback.strip()
                    if cleaned.startswith("```"):
                        cleaned = cleaned.split("\n", 1)[1]
                    if cleaned.endswith("```"):
                        cleaned = cleaned.rsplit("\n", 1)[0]
                    critique_feedback = json.loads(cleaned)
                    print(f"[ORCHESTRATOR DEBUG] Parsed critique feedback: {critique_feedback}")
                except Exception as e:
                    print(f"[ORCHESTRATOR DEBUG] Failed to parse critique feedback: {e}")
                    critique_feedback = {}

            if critique_feedback:
                direct_rating = critique_feedback.get('rating', 'NOT FOUND')
                print(f"[ORCHESTRATOR DEBUG] Direct rating from critique_feedback: {direct_rating}")

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
            parsed_rating = 0.0
            critique_details_for_history = None # This will hold the structured critique

            if isinstance(critique_feedback_raw, str):
                raw_critique_str_for_parsing = critique_feedback_raw # Original raw string
                
                # Strip Markdown fences if present
                processed_str = raw_critique_str_for_parsing.strip()
                # Remove leading ```json or ```
                if processed_str.startswith("```json"):
                    processed_str = processed_str[len("```json"):]
                elif processed_str.startswith("```"): # Handle case where it might just be ```
                    processed_str = processed_str[len("```"):]
                
                # Remove trailing ```
                if processed_str.endswith("```"):
                    processed_str = processed_str[:-len("```")]
                
                # Strip again to clean up any whitespace left by fence removal or originally present
                processed_str = processed_str.strip()

                try:
                    critique_feedback_parsed = json.loads(processed_str) # Use processed_str for parsing
                    rating_value = critique_feedback_parsed.get("rating")
                    if rating_value is not None:
                        try:
                            rating = float(rating_value)
                        except (ValueError, TypeError):
                            logger.warning(f"[{self.name}] Could not convert rating '{rating_value}' from parsed JSON to float. Defaulting to 0.0.")
                            rating = 0.0
                    else:
                        rating = 0.0 # Default if 'rating' key is missing in parsed JSON
                    
                    critique_details_for_history = critique_feedback_parsed

                except json.JSONDecodeError as e:
                    logger.error(f"[{self.name}] Failed to parse critique string: {e}. Original Raw: '{raw_critique_str_for_parsing}'. Processed attempt: '{processed_str}'")
                    # Rating remains 0.0 (default)
                    critique_details_for_history = {"error": f"JSONDecodeError: {e}", "original_raw": raw_critique_str_for_parsing, "processed_attempt": processed_str}
                except Exception as e: # Catch other potential errors
                    logger.error(f"[{self.name}] An unexpected error occurred while processing string critique_feedback: {e}. Original Raw: '{raw_critique_str_for_parsing}'. Processed attempt: '{processed_str}'")
                    critique_details_for_history = {"error": f"Unexpected error: {e}", "original_raw": raw_critique_str_for_parsing, "processed_attempt": processed_str}
            elif isinstance(critique_feedback_raw, dict):
                critique_details_for_history = critique_feedback_raw
                rating = float(critique_feedback_raw.get("rating", 0.0))
            else:
                logger.warning(f"[{self.name}] critique_feedback_raw is not a string or dict: {type(critique_feedback_raw)}. Raw: {critique_feedback_raw}")
                critique_details_for_history = {"error": "Critique was not string or dict", "raw": str(critique_feedback_raw)}

            logger.info(f"[{self.name}] Iteration {iteration}: Critique processed, parsed_rating = {parsed_rating}")
            print(f"[ORCHESTRATOR DEBUG] Parsed rating from critique_feedback: {parsed_rating}")
            print(f"[ORCHESTRATOR DEBUG] Critique details for history: {critique_details_for_history}")

            # Store the current iteration's answer, critique, and rating
            current_star_answer_for_history = ctx.session.state.get("answer") 
            
            iteration_entry = {
                "iteration_number": iteration,
                "answer": current_star_answer_for_history,
                "critique": critique_details_for_history, 
                "rating": rating
            }
            
            current_history_list = ctx.session.state.get("full_iteration_history", [])
            if not isinstance(current_history_list, list):
                logger.warning(f"[{self.name}] 'full_iteration_history' was not a list during append. Resetting.")
                current_history_list = []
            
            new_history_list = current_history_list + [iteration_entry]

            if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                ctx.actions.state_delta = {"full_iteration_history": new_history_list}
                ctx.session.state["full_iteration_history"] = new_history_list
            else:
                ctx.session.state["full_iteration_history"] = new_history_list
            
            logger.info(f"[{self.name}] Added iteration {iteration} details to full_iteration_history.")
            # Optional: More detailed debug log for the appended item
            if new_history_list:
                last_entry = new_history_list[-1]
                print(f"[ORCHESTRATOR DEBUG] Last item in full_iteration_history: iteration_number={last_entry.get('iteration_number')}, rating={last_entry.get('rating')}, answer_keys_present={list(last_entry.get('answer').keys()) if isinstance(last_entry.get('answer'), dict) else type(last_entry.get('answer'))}, critique_keys_present={list(last_entry.get('critique').keys()) if isinstance(last_entry.get('critique'), dict) else type(last_entry.get('critique'))}")
            else:
                print("[ORCHESTRATOR DEBUG] full_iteration_history is empty after trying to append.")

            final_rating = rating # Update final_rating with the latest one, to be used for threshold check
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
                # Update state with error status
                if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
                    state_delta = {
                        "final_status": "ERROR_AGENT_PROCESSING",
                        "error_message": f"Star refiner failed: {str(e)}"
                    }
                    ctx.actions.state_delta = state_delta
                else:
                    ctx.session.state["final_status"] = "ERROR_AGENT_PROCESSING"
                    ctx.session.state["error_message"] = f"Star refiner failed: {str(e)}"

                # Run output retriever to format error
                async for event in self.output_retriever.run_async(ctx):
                    yield event
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

        # Step 6: Retrieve and format final output (now includes timing data)
        logger.info(f"[{self.name}] Retrieving final output...")
        with time_operation(self.timing_tracker, "output_retriever"):
            async for event in self.output_retriever.run_async(ctx):
                # Log event information for debugging
                logger.info(f"[{self.name}] Event from output_retriever: {event.author} (has_content={event.content is not None})")
                # Forward events from the output_retriever
                yield event

        # Update timing with output retriever time and log final report
        final_timing_data = self.timing_tracker.get_timings()
        if hasattr(ctx, 'actions') and hasattr(ctx.actions, 'state_delta'):
            ctx.actions.state_delta = {"timing_data": final_timing_data}
        else:
            ctx.session.state["timing_data"] = final_timing_data

        # Log timing report
        from .timing import format_timing_report
        timing_report = format_timing_report(timing_data)
        logger.info(f"\n{timing_report}")

        logger.info(f"[{self.name}] STAR workflow completed with status: {ctx.session.state.get('final_status')}")


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