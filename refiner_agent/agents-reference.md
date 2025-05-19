# STAR Refiner Agents Reference Guide

This document provides a comprehensive overview of all agents in the STAR Refiner system, including their configuration, purpose, tools, and models.

## Root Orchestrator Agent

| Property | Value |
|----------|-------|
| **Name** | `refiner_agent` |
| **Type** | `STAROrchestrator` (custom `BaseAgent`) |
| **Description** | Custom orchestrator for STAR format answer generation with conditional refinement |
| **Location** | `/orchestrator.py` |
| **Sub-agents** | InitializeHistoryAgent, InputCollector, STARGeneratorWithHistory, STARCritiqueWithHistory, STARRefinerWithHistory, FinalOutputRetrieverAgent |
| **Configuration** | Rating threshold: 4.6 (configurable), Max iterations: 3 (configurable) |

The orchestrator is responsible for coordinating the entire workflow, managing state, and making decisions about when to continue or stop the refinement process. It directly parses and processes agent outputs, maintains the `full_iteration_history`, and yields the final event with the complete response.

## Sub-agents

### 1. Initialize History Agent

| Property | Value |
|----------|-------|
| **Name** | `InitializeHistoryAgent` |
| **Type** | `Agent` |
| **Description** | Initializes history tracking for STAR responses and critiques |
| **Model** | `gemini-2.0-flash` (configurable via `INITIALIZE_AGENT_MODEL`) |
| **Tools** | `initialize_history` |
| **Output Key** | None |
| **Location** | `/agent.py` |

This agent sets up the initial state structures needed for the STAR workflow, particularly the `full_iteration_history` array.

### 2. Input Collector Agent

| Property | Value |
|----------|-------|
| **Name** | `InputCollector` |
| **Type** | `LlmAgent` |
| **Description** | Collects the required information for generating STAR format answers |
| **Model** | `gemini-2.0-flash` (configurable via `INPUT_COLLECTOR_MODEL`) |
| **Tools** | `collect_star_inputs` |
| **Output Key** | `input_data` |
| **Location** | `/subagents/input_collector/agent.py` |

This agent processes and validates the input parameters (role, industry, question) and prepares them for the generator agent.

### 3. STAR Generator Agent

| Property | Value |
|----------|-------|
| **Name** | `STARGeneratorWithHistory` (wraps `STARAnswerGenerator`) |
| **Type** | `Agent` (wrapper) / `LlmAgent` (base) |
| **Description** | Generates initial STAR format answers for interview questions |
| **Model** | `gemini-2.0-flash` (configurable via `STAR_GENERATOR_MODEL`) |
| **Tools** | None |
| **Output Key** | `current_answer` |
| **Location** | `/agent.py` (wrapper), `/subagents/generator/agent.py` (base) |

This agent generates the initial STAR-formatted answer based on the user's role, industry, and question. The orchestrator parses its output and adds it to the `full_iteration_history`.

### 4. STAR Critique Agent

| Property | Value |
|----------|-------|
| **Name** | `STARCritiqueWithHistory` (wraps `STARAnswerCritic`) |
| **Type** | `Agent` (wrapper) / `LlmAgent` (base) |
| **Description** | Evaluates STAR answers and provides specific feedback for improvement |
| **Model** | `gemini-2.0-flash` (configurable via `STAR_CRITIQUE_MODEL`) |
| **Tools** | `rate_star_answer` |
| **Output Key** | `critique_feedback` |
| **Location** | `/agent.py` (wrapper), `/subagents/critique/agent.py` (base) |

This agent analyzes the STAR answer, provides a detailed critique, and assigns a rating on a scale of 1.0 to 5.0. The orchestrator parses its output and adds the critique to the corresponding entry in `full_iteration_history`.

### 5. STAR Refiner Agent

| Property | Value |
|----------|-------|
| **Name** | `STARRefinerWithHistory` (wraps `STARAnswerRefiner`) |
| **Type** | `Agent` (wrapper) / `LlmAgent` (base) |
| **Description** | Refines STAR format answers based on specific critique feedback |
| **Model** | `gemini-2.0-flash` (configurable via `STAR_REFINER_MODEL`) |
| **Tools** | None |
| **Output Key** | `current_answer` |
| **Location** | `/agent.py` (wrapper), `/subagents/refiner/agent.py` (base) |

This agent takes the original STAR answer and critique feedback, then generates an improved version of the answer. The orchestrator parses its output and adds it as a new entry in `full_iteration_history`.

### 6. Final Output Retriever Agent

| Property | Value |
|----------|-------|
| **Name** | `FinalOutputRetrieverAgent` |
| **Type** | `LlmAgent` |
| **Description** | Retrieves the final structured output from session state |
| **Model** | `gemini-2.0-flash` (configurable via `OUTPUT_RETRIEVER_MODEL`) |
| **Tools** | None |
| **Output Key** | `final_structured_output_from_state` |
| **Location** | `/agent.py` |

This agent is designed to retrieve the final output, but in the current implementation, the orchestrator bypasses this agent and directly calls the `retrieve_final_output_from_state` function to create the final JSON response.

## Agent Distribution by Tools

### Agents with Tools

1. **InitializeHistoryAgent**
   - Tool: `initialize_history`

2. **InputCollector**
   - Tool: `collect_star_inputs`

3. **STARCritiqueWithHistory**
   - Tool: `rate_star_answer`

### Agents without Tools

1. **STARGeneratorWithHistory** / **STARAnswerGenerator**
   - No tools: The orchestrator parses its output directly

2. **STARRefinerWithHistory** / **STARAnswerRefiner**
   - No tools: The orchestrator parses its output directly

3. **FinalOutputRetrieverAgent**
   - No tools: Recent refactoring bypasses this agent

## Notes on Recent Refactoring

The system has recently undergone refactoring to simplify tool usage:

1. Previously, agents like STARGeneratorWithHistory and STARRefinerWithHistory may have used tools like `append_star_answer` to update the state.

2. In the current implementation, the orchestrator directly parses the agent outputs and maintains the `full_iteration_history`.

3. The `FinalOutputRetrieverAgent` is now largely bypassed, with the orchestrator directly calling the `retrieve_final_output_from_state` function.

This centralized approach simplifies the workflow and reduces the number of required tool calls.

## Configuration

All agent models are configurable through environment variables or the `config.py` file:

```python
DEFAULT_MODELS = {
    "STAR_GENERATOR_MODEL": "gemini-2.0-flash",
    "STAR_CRITIQUE_MODEL": "gemini-2.0-flash",
    "STAR_REFINER_MODEL": "gemini-2.0-flash",
    "INPUT_COLLECTOR_MODEL": "gemini-2.0-flash",
    "INITIALIZE_AGENT_MODEL": "gemini-2.0-flash",
    "OUTPUT_RETRIEVER_MODEL": "gemini-2.0-flash",
}
```

Other configurable parameters:
- `RATING_THRESHOLD`: 4.6 (default)
- `MAX_ITERATIONS`: 3 (default)