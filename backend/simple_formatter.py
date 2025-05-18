"""
Simple Formatter

A clean, minimal formatter that just displays the final STAR answer.
This approach focuses on the core functionality without complex history extraction logic.
"""

from typing import Dict, Any, List, Optional

def format_simple_response(final_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Formats a clean, simple response with just the essential information.
    
    Args:
        final_output: The final output from the agent
        
    Returns:
        A formatted response with the STAR answer and feedback
    """
    # Initialize response with sensible defaults
    formatted_response = {
        "star_answer": None,
        "feedback": {
            "rating": 0.0,
            "suggestions": []
        },
        "history": [],
        "metadata": {
            "status": "COMPLETED",
            "highest_rating": 0.0,
            "role": None,
            "industry": None,
            "question": None
        }
    }
    
    # Extract STAR answer (most important part)
    if "answer" in final_output and isinstance(final_output["answer"], dict):
        formatted_response["star_answer"] = final_output["answer"]
    
    # Extract rating
    if "rating" in final_output:
        try:
            numeric_rating = float(final_output["rating"])
            formatted_response["feedback"]["rating"] = numeric_rating
            formatted_response["metadata"]["highest_rating"] = numeric_rating
        except (ValueError, TypeError):
            pass
            
    # For history, we'll use a simple approach:
    # Check for both "history" and "histories" keys
    history_list = []

    # Check for "history" key first (original format)
    if "history" in final_output and isinstance(final_output["history"], list):
        history_list = final_output["history"]
    # Then check for "histories" key (new format)
    elif "histories" in final_output and isinstance(final_output["histories"], list):
        history_list = final_output["histories"]

    # Process history items
    for i, history_item in enumerate(history_list):
        # Handle different history formats
        if isinstance(history_item, dict):
            # Case 1: Item has direct rating property
            if "rating" in history_item:
                simple_item = {
                    "iteration": i + 1,
                    "star_answer": formatted_response["star_answer"],  # Use the main answer
                    "critique": {
                        "rating": history_item.get("rating", 0.0),
                        "suggestions": history_item.get("suggestions", [])
                    }
                }
                formatted_response["history"].append(simple_item)

            # Case 2: Item has nested critique object (most common format)
            elif "critique" in history_item and isinstance(history_item["critique"], dict):
                # Get answer from history if available, otherwise use main answer
                answer_obj = history_item.get("answer", formatted_response["star_answer"])
                critique_obj = history_item["critique"]

                simple_item = {
                    "iteration": i + 1,
                    "star_answer": answer_obj,
                    "critique": {
                        "rating": critique_obj.get("rating", 0.0),
                        "suggestions": critique_obj.get("suggestions", [])
                    }
                }
                formatted_response["history"].append(simple_item)
                
    # Return the clean, simple response
    return formatted_response