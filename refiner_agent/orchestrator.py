"""
STAR Answer Orchestrator Agent

This module defines a custom orchestrator agent that manages the STAR answer generation workflow
with precise control over the refinement process.
"""

import logging
import json
from typing import AsyncGenerator, Optional
from typing_extensions import override

from google.adk.agents import Agent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

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
        logger.info(f"[{self.name}] Starting STAR answer generation workflow.")
        
        # Step 1: Initialize history
        logger.info(f"[{self.name}] Initializing history...")
        async for event in self.initialize_agent.run_async(ctx):
            # Log event information for debugging
            logger.info(f"[{self.name}] Event from initialize_agent: {event.author} (has_content={event.content is not None})")
            # Forward events from the initialize_agent
            yield event
        
        # Step 2: Collect inputs
        logger.info(f"[{self.name}] Collecting inputs...")
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
        try:
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
            logger.info(f"[{self.name}] Starting iteration {iteration}")
            
            # Run critique
            logger.info(f"[{self.name}] Running critique...")
            try:
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

            # Get the rating from the current iteration
            rating = 0.0
            iterations = current_session.state.get("iterations", [])
            current_iteration = current_session.state.get("current_iteration", 1)

            # Find the current iteration's critique
            for iter_data in iterations:
                if iter_data.get("iteration") == current_iteration:
                    critique = iter_data.get("critique", {})
                    rating = float(critique.get("rating", 0.0))
                    logger.info(f"[{self.name}] Retrieved rating {rating} from iteration {current_iteration}")
                    break

            final_rating = rating
            highest_rating = current_session.state.get("highest_rating", 0.0)
            logger.info(f"[{self.name}] Critique rating: {rating}, highest rating: {highest_rating}")
            # Note: highest_rating and highest_rated_iteration are managed by append_critique tool
            
            # Update iteration state
            # Use state delta to ensure atomic updates
            update_iteration_info(ctx, iteration)
            
            # Use the current rating for threshold check
            threshold_check_rating = rating
            logger.info(f"[{self.name}] Current rating: {rating}, Highest rating: {highest_rating}")
            logger.info(f"[{self.name}] Using rating {threshold_check_rating} for threshold check (threshold: {self.rating_threshold})")

            # Check if rating meets threshold to skip refinement
            if threshold_check_rating >= self.rating_threshold:
                logger.info(f"[{self.name}] Rating {threshold_check_rating} meets threshold {self.rating_threshold}. Stopping refinement.")

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
            try:
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
            
            # Update iteration counter
            iteration += 1
        
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
        
        # Step 5: Retrieve and format final output
        logger.info(f"[{self.name}] Retrieving final output...")
        async for event in self.output_retriever.run_async(ctx):
            # Log event information for debugging
            logger.info(f"[{self.name}] Event from output_retriever: {event.author} (has_content={event.content is not None})")
            # Forward events from the output_retriever
            yield event
        
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