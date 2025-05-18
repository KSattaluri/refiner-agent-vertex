"""
Consistent Object Handling Utilities

This module provides utility functions for handling objects consistently
throughout the application, especially for dealing with empty objects and
None values.
"""

import json
from typing import Dict, Any, Optional, List, Union, Type
from pydantic import BaseModel


def clean_json_string(json_string: str) -> str:
    """
    Clean JSON strings from markdown formatting.

    Args:
        json_string: The JSON string to clean

    Returns:
        The cleaned JSON string
    """
    if not isinstance(json_string, str):
        return ""
        
    cleaned_json = json_string
    if cleaned_json.startswith("```json"):
        cleaned_json = cleaned_json[len("```json"):].strip()
    elif cleaned_json.startswith("```"):
        cleaned_json = cleaned_json[len("```"):].strip()
    if cleaned_json.endswith("```"):
        cleaned_json = cleaned_json[:-len("```")].strip()

    return cleaned_json


def is_empty_object(obj: Any) -> bool:
    """
    Check if an object is effectively empty.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is None, an empty dict, an empty list, or a dict with all None values
    """
    if obj is None:
        return True
        
    if isinstance(obj, dict):
        if len(obj) == 0:
            return True
        # Check if all values are None or empty objects themselves
        return all(is_empty_object(v) for v in obj.values())
        
    if isinstance(obj, list):
        return len(obj) == 0
        
    if isinstance(obj, str):
        return obj.strip() == ""
        
    # For numbers, booleans, and other types, they're not considered empty
    return False


def ensure_dict_or_none(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Ensure an object is either a dictionary or None.
    
    Args:
        obj: The object to process
        
    Returns:
        A dictionary, None, or a dictionary representation of the object
    """
    if obj is None:
        return None
        
    if is_empty_object(obj):
        return None
        
    if isinstance(obj, dict):
        return obj
        
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
        
    # Handle string JSON
    if isinstance(obj, str):
        try:
            cleaned = clean_json_string(obj)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If it's not valid JSON, create a minimal dict with the string as content
            return {"content": obj[:200] + "..." if len(obj) > 200 else obj}
    
    # For other types, try to convert to dict if possible
    try:
        return dict(obj)
    except (TypeError, ValueError):
        # For anything else, create a minimal dict with str representation
        return {"content": str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)}


def safe_get_value(obj: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary that may be None.
    
    Args:
        obj: The dictionary to get the value from
        key: The key to get
        default: The default value to return if the key doesn't exist
        
    Returns:
        The value for the key, or the default if the key doesn't exist or obj is None
    """
    if obj is None:
        return default
        
    return obj.get(key, default)


def convert_to_model_or_none(obj: Any, model_class: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Convert an object to a Pydantic model instance or None.
    
    Args:
        obj: The object to convert
        model_class: The Pydantic model class to use
        
    Returns:
        A model instance or None
    """
    if obj is None:
        return None
        
    if is_empty_object(obj):
        return None
        
    # If it's already an instance of the model, return it
    if isinstance(obj, model_class):
        return obj
        
    try:
        # Try to validate directly
        return model_class.model_validate(obj)
    except Exception:
        try:
            # If it's a string, try to parse as JSON first
            if isinstance(obj, str):
                cleaned = clean_json_string(obj)
                parsed = json.loads(cleaned)
                return model_class.model_validate(parsed)
        except Exception:
            # If all else fails, return None
            return None
            
    return None


def standardize_object(
    obj: Any, 
    allow_none: bool = True, 
    empty_to_none: bool = True
) -> Union[Dict[str, Any], List[Any], None]:
    """
    Convert various object types to a standardized format.
    
    This is the main entry point for object standardization in the application.
    
    Args:
        obj: The object to standardize
        allow_none: Whether to allow returning None
        empty_to_none: Whether to convert empty objects to None
        
    Returns:
        A standardized object (dict, list) or None
    """
    # Handle empty cases first
    if obj is None and allow_none:
        return None
        
    if empty_to_none and is_empty_object(obj):
        return None if allow_none else {}
        
    # Handle dictionaries
    if isinstance(obj, dict):
        # Process recursively for nested objects
        result = {}
        for k, v in obj.items():
            processed_value = standardize_object(v, allow_none=True, empty_to_none=empty_to_none)
            if processed_value is not None or allow_none:
                result[k] = processed_value
        return result if result or not empty_to_none else None
        
    # Handle lists
    if isinstance(obj, list):
        result = [
            standardize_object(item, allow_none=True, empty_to_none=empty_to_none) 
            for item in obj
        ]
        # Filter out None values if not allowed
        if not allow_none:
            result = [item for item in result if item is not None]
        return result if result or not empty_to_none else None
        
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        return standardize_object(
            obj.model_dump(), 
            allow_none=allow_none, 
            empty_to_none=empty_to_none
        )
        
    # Handle string JSON
    if isinstance(obj, str):
        try:
            cleaned = clean_json_string(obj)
            parsed = json.loads(cleaned)
            return standardize_object(
                parsed, 
                allow_none=allow_none, 
                empty_to_none=empty_to_none
            )
        except json.JSONDecodeError:
            # It's just a regular string, pass through
            return obj
            
    # For primitive types (int, float, bool, etc.), pass through
    return obj