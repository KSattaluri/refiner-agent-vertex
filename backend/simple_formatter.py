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
        final_output: The final output from the agent - should have 'answer', 'history', and 'rating'
        
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
    
    # The history should always be under "history" key with items containing "answer" and "critique"
    if "history" in final_output and isinstance(final_output["history"], list):
        history_list = final_output["history"]
        
        for i, history_item in enumerate(history_list):
            if isinstance(history_item, dict) and "answer" in history_item and "critique" in history_item:
                simple_item = {
                    "iteration": i + 1,
                    "star_answer": history_item["answer"],
                    "critique": history_item["critique"]
                }
                formatted_response["history"].append(simple_item)
            else:
                print(f"[DEBUG] Unexpected history item format at index {i}: {history_item}")
                
    # Return the clean, simple response
    return formatted_response