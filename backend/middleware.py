"""
Middleware for the STAR Answer Generator API.

This module provides middleware functions for request/response handling,
including validation, error handling, and logging.
"""

import json
import logging
import functools
import traceback
from typing import Callable, Type, Dict, Any, List, Optional

from flask import request, jsonify
from pydantic import BaseModel, ValidationError

from .validation import ErrorResponse, ValidationError as APIValidationError

# Configure logging
logger = logging.getLogger(__name__)

def validate_request(model: Type[BaseModel]) -> Callable:
    """
    Decorator for validating API request bodies against a Pydantic model.
    
    Args:
        model: The Pydantic model class to validate against
        
    Returns:
        A decorator function that validates the request body
    
    Example:
        @app.route('/api/resource', methods=['POST'])
        @validate_request(ResourceModel)
        def create_resource(validated_data):
            # validated_data is a ResourceModel instance
            pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Extract request body
            request_data = None
            validation_errors = []
            
            try:
                request_data = request.get_json(silent=True)
                
                # Check if request body is present
                if request_data is None:
                    validation_errors.append(
                        APIValidationError(field="request", message="Missing or invalid JSON body")
                    )
                    return handle_validation_errors(validation_errors)
                
                # Validate against Pydantic model
                try:
                    validated_data = model.model_validate(request_data)
                    # Call the original function with validated data
                    return f(validated_data, *args, **kwargs)
                except ValidationError as e:
                    # Convert Pydantic validation errors to our format
                    for error in e.errors():
                        field = ".".join(str(loc) for loc in error["loc"])
                        message = error["msg"]
                        validation_errors.append(APIValidationError(field=field, message=message))

                    logger.error(f"Validation errors: {validation_errors}")
                    return handle_validation_errors(validation_errors)
            except Exception as e:
                logger.error(f"Unexpected error in validation middleware: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Generic error response
                error_response = ErrorResponse(
                    metadata={
                        "status": "ERROR_VALIDATION",
                        "error_message": "An error occurred while validating the request"
                    }
                )
                return jsonify(error_response.model_dump()), 400
                
        return wrapper
    return decorator

def handle_validation_errors(errors: List[APIValidationError]) -> tuple:
    """
    Creates a standardized error response for validation errors.
    
    Args:
        errors: List of validation error objects
        
    Returns:
        A tuple containing the JSON response and HTTP status code
    """
    # Format validation errors
    formatted_errors = [error.model_dump() for error in errors]
    
    # Create error response
    error_response = ErrorResponse(
        metadata={
            "status": "ERROR_VALIDATION",
            "error_message": "Validation failed for the request data"
        },
        validation_errors=formatted_errors
    )
    
    # Return formatted response
    return jsonify(error_response.model_dump()), 422  # 422 Unprocessable Entity

def validate_response(response_data: Dict[str, Any], model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Validates API responses against a Pydantic model.
    
    Args:
        response_data: The data to validate
        model: The Pydantic model class to validate against
        
    Returns:
        The validated data or a fallback error response
    """
    try:
        # Validate against the model
        validated = model.model_validate(response_data)
        return validated.model_dump()
    except ValidationError as e:
        logger.error(f"Response validation error: {e}")
        
        # Create a fallback response
        error_metadata = {
            "status": "ERROR_RESPONSE_VALIDATION",
            "error_message": "The system generated a malformed response"
        }
        
        # Try to preserve original metadata if possible
        if isinstance(response_data, dict) and "metadata" in response_data:
            if isinstance(response_data["metadata"], dict) and "status" in response_data["metadata"]:
                error_metadata["status"] = response_data["metadata"]["status"]
            
        # Create minimal valid response that matches our updated validators
        fallback_response = {
            "star_answer": None,  # Make this optional
            "feedback": None,  # Make this optional
            "history": [],
            "metadata": error_metadata
        }
        
        return fallback_response