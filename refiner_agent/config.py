"""
Configuration settings for the Refiner Agent

This module provides centralized configuration for model names and other settings.
Models can be configured through environment variables or use defaults.
"""

import os
from typing import Dict

# Default model configurations
DEFAULT_MODELS = {
    "STAR_GENERATOR_MODEL": "gemini-2.0-flash",
    "STAR_CRITIQUE_MODEL": "gemini-2.0-flash",
    "STAR_REFINER_MODEL": "gemini-2.0-flash",
    "INPUT_COLLECTOR_MODEL": "gemini-2.0-flash",
    "INITIALIZE_AGENT_MODEL": "gemini-2.0-flash",
    "OUTPUT_RETRIEVER_MODEL": "gemini-2.0-flash",
}

# Load model configurations from environment or use defaults
MODELS: Dict[str, str] = {}
for key, default_value in DEFAULT_MODELS.items():
    MODELS[key] = os.getenv(key, default_value)

# For backward compatibility, expose individual constants
STAR_GENERATOR_MODEL = MODELS["STAR_GENERATOR_MODEL"]
STAR_CRITIQUE_MODEL = MODELS["STAR_CRITIQUE_MODEL"]
STAR_REFINER_MODEL = MODELS["STAR_REFINER_MODEL"]
INPUT_COLLECTOR_MODEL = MODELS["INPUT_COLLECTOR_MODEL"]
INITIALIZE_AGENT_MODEL = MODELS["INITIALIZE_AGENT_MODEL"]
OUTPUT_RETRIEVER_MODEL = MODELS["OUTPUT_RETRIEVER_MODEL"]

# Other configuration settings
RATING_THRESHOLD = float(os.getenv("RATING_THRESHOLD", "4.6"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))