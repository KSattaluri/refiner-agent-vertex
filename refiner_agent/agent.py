"""
STAR Answer Generator with Custom Orchestration

This module defines the root agent for the STAR answer generation application.
It uses a custom orchestrator that provides precise control over the refinement process.
"""

from google.adk.agents import Agent
from google.adk.agents.llm_agent import LlmAgent

# Import subagents
from .subagents.input_collector.agent import input_collector
from .subagents.generator.agent import star_generator
from .subagents.critique.agent import star_critique
from .subagents.refiner.agent import star_refiner

# Import tools
from .tools import (
    initialize_history, 
    append_star_response, 
    append_critique, 
    rate_star_answer, 
    retrieve_final_output_from_state
)

# Import custom orchestrator
from .orchestrator import STAROrchestrator

# Create a simple history initialization agent
initialize_agent = Agent(
    name="InitializeHistoryAgent",
    model="gemini-2.0-flash",
    description="Initializes history tracking for STAR responses and critiques",
    tools=[initialize_history]
)

# Modify star_generator to handle appending responses
star_generator_with_history = Agent(
    name="STARGeneratorWithHistory",
    model=star_generator.model,
    description=star_generator.description,
    instruction=star_generator.instruction,
    tools=[append_star_response],  # Add tool to append to history
    output_key=star_generator.output_key
)

# Modify star_critique to handle appending critiques
# Note: No need for exit_refinement_loop tool since orchestrator handles the flow
star_critique_with_history = Agent(
    name="STARCritiqueWithHistory",
    model=star_critique.model,
    description=star_critique.description,
    instruction="""You are a STAR Answer Quality Evaluator with EXCEPTIONALLY HIGH STANDARDS.

    Your task is to rigorously evaluate the quality of a STAR format interview answer and provide a stringent rating and detailed feedback.
    
    ## STAR ANSWER TO EVALUATE
    {current_answer}
    
    ## EVALUATION CRITERIA
    Rate the answer on a scale of 1.0 to 5.0 based on these criteria. Be STRICT - a perfect 5.0 should be extremely rare and reserved only for truly exceptional answers.
    
    1. **Structure** (25%):
       - Does it clearly follow the STAR format with distinct, well-developed sections?
       - Are all four components (Situation, Task, Action, Result) clearly identifiable and properly balanced?
       - Is the flow logical, coherent, and well-organized with smooth transitions?
       - Does it have appropriate length for each component (not too short or verbose)?
    
    2. **Relevance** (25%):
       - Is the answer precisely tailored to the specific role and industry context?
       - Does it address the question directly and comprehensively?
       - Does it highlight skills and experiences that are highly relevant to the position?
       - Is the example chosen particularly appropriate for the question asked?
    
    3. **Specificity** (25%):
       - Does it use precise, concrete examples with specific details, names, dates, and metrics?
       - Are there multiple quantifiable results with meaningful metrics?
       - Does it completely avoid vague generalities, clichés, and generic statements?
       - Does it provide rich context that makes the story compelling and believable?
    
    4. **Professional Impact** (25%):
       - Is the tone consistently professional, confident, and appropriate throughout?
       - Does it effectively showcase the candidate's unique abilities, leadership, and initiative?
       - Is it concise yet comprehensive, with no unnecessary details?
       - Does it demonstrate significant value and impact through the candidate's actions?
    
    ## RATING CALCULATION (MANDATORY METHOD)
    You MUST follow this exact calculation method:

    1. Rate each of the four criteria separately on a 1-5 scale using these standards:
       - Structure (1-5): If any STAR component is missing or unclear, maximum score is 3.0
       - Relevance (1-5): If not specifically tailored to the role/industry, maximum score is 3.5
       - Specificity (1-5): If lacking concrete metrics or dates, maximum score is 3.0
       - Professional Impact (1-5): If using generic phrases without evidence, maximum score is 3.5

    2. Apply these automatic deductions:
       - No specific company or project name mentioned: -0.3 points
       - No specific metrics in results: -0.5 points
       - No specific timeframe mentioned: -0.3 points
       - Generic or clichéd language: -0.4 points
       - Imbalanced section lengths: -0.2 points

    3. Calculate the final score:
       - Start with the average of the four criteria scores
       - Apply all applicable automatic deductions
       - Round to the nearest 0.1

    ## RATING GUIDELINES
    - 5.0: Exceptional, nearly flawless answer (should almost never be given)
    - 4.6-4.9: Excellent answer with minimal improvements needed (rare, <5% of answers)
    - 4.0-4.5: Good answer but has clear improvements needed (uncommon, ~15% of answers)
    - 3.0-3.9: Average answer with significant shortcomings (most common, ~60% of answers)
    - 2.0-2.9: Below average answer with major deficiencies (~15% of answers)
    - 1.0-1.9: Unacceptable answer with fundamental problems (~5% of answers)

    BE EXTREMELY STRICT WITH RATINGS:
    - First-time answers should almost never exceed 4.3
    - Most answers should fall between 3.0-4.0
    - ANY answer lacking specific metrics or concrete details CANNOT score above 4.0
    - ANY answer using generic business language without specific examples CANNOT score above 3.8
    - MUST follow the mandatory calculation method above
    
    ## OUTPUT INSTRUCTIONS
    
    1. Call the `rate_star_answer` tool to analyze the answer.
    2. Call the `append_critique` tool with your detailed feedback.
    3. Return your critique as JSON output with the following keys:
       - "rating": A float representing the overall numerical rating (e.g., 4.2)
       - "structure_feedback": Specific feedback on the answer's structure
       - "relevance_feedback": Specific feedback on the answer's relevance
       - "specificity_feedback": Specific feedback on the answer's specificity
       - "professional_impact_feedback": Specific feedback on the answer's professional impact
       - "suggestions": A list of 2-3 strings, each a concrete suggestion for improvement
    """,
    tools=[rate_star_answer, append_critique],
    output_key=star_critique.output_key
)

# Modify star_refiner to handle appending refined responses
star_refiner_with_history = Agent(
    name="STARRefinerWithHistory",
    model=star_refiner.model,
    description=star_refiner.description,
    instruction=star_refiner.instruction,
    tools=[append_star_response],
    output_key=star_refiner.output_key
)

# Agent for retrieving final output from state
final_output_retriever = LlmAgent(
    name="FinalOutputRetrieverAgent",
    description="Retrieves the final structured output from session state and formats it as the definitive response.",
    model="gemini-2.0-flash", 
    tools=[retrieve_final_output_from_state],
    instruction="""Your sole task is to retrieve the final agent output package from the session state.
You MUST call the `retrieve_final_output_from_state` tool.
Then, you MUST output the exact JSON string you receive from the tool, wrapped in markdown ```json ... ``` fences.
DO NOT add any other explanatory text or conversation. Only the JSON string in markdown.
The value associated with 'retrieved_output' is a dictionary containing the complete final data (STAR answer, histories, rating).
You MUST take this 'retrieved_output' dictionary and output it as your final response.
Your entire output MUST be ONLY this dictionary, formatted as a JSON string, and wrapped in markdown code fences (```json ... ```).
Do not add any other text, explanations, or conversational filler.""",
    output_key="final_structured_output_from_state"
)

# Create the custom orchestrator as the root agent
root_agent = STAROrchestrator(
    name="refiner_agent",  # IMPORTANT: This MUST match the directory name
    initialize_agent=initialize_agent,
    input_collector=input_collector,
    star_generator=star_generator_with_history,
    star_critique=star_critique_with_history,
    star_refiner=star_refiner_with_history,
    output_retriever=final_output_retriever,
    rating_threshold=4.6,  # Skip refinement when rating is at least 4.6
    max_iterations=3       # Maximum of 3 refinement iterations
)