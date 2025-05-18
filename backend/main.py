"""
Flask API Server for STAR Answer Generator Agent

This module provides a web interface for interacting with the STAR answer generator agent.
It creates a single HTTP endpoint (/chat) that accepts interview questions and returns
structured STAR format answers with critiques and refinement history.
"""

from flask import Flask, request, jsonify, session as flask_session, send_from_directory
from .validation import STARGeneratorRequest, STARGeneratorResponse, LLMPromptData
from .middleware import validate_request, validate_response
from .simple_formatter import format_simple_response
import os
import json
import uuid
import traceback
import datetime
from dotenv import load_dotenv
import vertexai
from vertexai.preview import reasoning_engines
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.genai.types import Content, Part
from refiner_agent.agent import root_agent
from refiner_agent.schemas import AgentFinalOutput, EnhancedAgentFinalOutput

# Load environment variables from .env file
load_dotenv()

# Get Google Cloud Project and Location from environment
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# Initialize Vertex AI SDK
if PROJECT_ID and LOCATION:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
else:
    print("Error: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env")

# Create session service for state management
session_service = InMemorySessionService()

# Create an instance of the ADK application
# Commented out for now due to import compatibility issues
# adk_instance = reasoning_engines.AdkApp(agent=root_agent)

# Define a constant app name for sessions
APP_NAME = "star_answer_generator"

# Create a Runner with the session service
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# Create Flask application instance
app = Flask(__name__)

# Set a secret key for Flask session management
app.secret_key = os.urandom(24)

def update_session_state(session, request_details, default_status="IN_PROGRESS"):
    """
    Updates session state with request details and default status.

    Args:
        session: The session object to update
        request_details: Dictionary containing request details
        default_status: Default status to set
    """
    # Update session state with request details
    for key, value in request_details.items():
        session.state[key] = value

    # Set default status
    session.state["final_status"] = default_status

    return session

def get_or_create_session(agent_session_id, user_id_for_agent, request_details, flask_session):
    """
    Gets an existing session or creates a new one with the provided request details.

    Args:
        agent_session_id: Existing session ID (may be None)
        user_id_for_agent: User identifier for the agent
        request_details: Dictionary containing request details
        flask_session: Flask session object for storing IDs

    Returns:
        Session object
    """
    session = None

    if not agent_session_id:
        # Create new session
        app.logger.info(f"Creating new agent session for user: {user_id_for_agent}")
        new_session_id = str(uuid.uuid4())

        # Create initial state with request details
        initial_state = request_details.copy()
        initial_state["final_status"] = "IN_PROGRESS"

        # Create session
        session = session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id_for_agent,
            session_id=new_session_id,
            state=initial_state
        )

        # Store session info in Flask session
        flask_session['agent_session_id'] = session.id
        flask_session['user_id_for_agent'] = user_id_for_agent

        app.logger.info(f"Created session with ID: {session.id}")
    else:
        # Get existing session
        session = session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id_for_agent,
            session_id=agent_session_id
        )

        # Update session state
        update_session_state(session, request_details)

        app.logger.info(f"Updated existing session {agent_session_id} with new request details")

    return session

# Configure logging
import logging

# Get logging level from environment, default to INFO for development
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

# Set Flask app logging level based on environment
if os.getenv('FLASK_ENV') == 'production':
    app.logger.setLevel(logging.WARNING)
else:
    app.logger.setLevel(logging.INFO)

# Suppress verbose Google ADK and related library logging
logging.getLogger('google.adk').setLevel(logging.WARNING)
logging.getLogger('google.adk.models').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('refiner_agent.orchestrator').setLevel(logging.WARNING)

# Global error handler for all routes
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions in a consistent format using event mechanism"""
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())

    error_message = f"Server error: {str(e)}"

    # Try to get session and append error event
    try:
        if 'agent_session_id' in flask_session and 'user_id_for_agent' in flask_session:
            user_id = flask_session.get('user_id_for_agent')
            session_id = flask_session.get('agent_session_id')

            # Get session
            session = session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id
            )

            if session:
                # Try to use EventActions for atomic updates if available
                event_actions = getattr(session, 'actions', None)

                if event_actions and hasattr(event_actions, 'state_delta'):
                    # Use state_delta for atomic updates
                    event_actions.state_delta = {
                        "final_status": "ERROR_SERVER",
                        "error_message": error_message,
                        "server_error": True
                    }
                    app.logger.info("Session state updated with error information via EventActions (global handler)")
                else:
                    # Fall back to direct state updates
                    session.state["final_status"] = "ERROR_SERVER"
                    session.state["error_message"] = error_message
                    session.state["server_error"] = True
                    app.logger.info("Session state updated with error information through global handler")
    except Exception as session_err:
        app.logger.error(f"Failed to handle session in global error handler: {session_err}")

    # Construct an error response in our formatted structure
    try:
        # Try to get request data for context
        request_data = request.get_json() if request.is_json else {}

        error_response = {
            "star_answer": None,
            "feedback": None,
            "history": [],
            "metadata": {
                "status": "ERROR_SERVER",
                "role": request_data.get("role", ""),
                "industry": request_data.get("industry", ""),
                "question": request_data.get("question", ""),
                "error_message": error_message
            }
        }
    except Exception:
        # Fallback with minimal info
        error_response = {
            "star_answer": None,
            "feedback": None,
            "history": [],
            "metadata": {
                "status": "ERROR_SERVER",
                "error_message": error_message
            }
        }

    return jsonify(error_response), 500

# Handle 404 errors
@app.route('/', methods=['GET'])
def serve_ui():
    """Serve the HTML UI for the STAR Answer Generator"""
    return send_from_directory('static', 'index.html')

@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors in a consistent format"""
    # Check if this is a request for the API endpoint
    if request.path.startswith('/chat'):
        error_response = {
            "star_answer": None,
            "feedback": None,
            "history": [],
            "metadata": {
                "status": "ERROR_NOT_FOUND",
                "error_message": "The requested endpoint was not found."
            }
        }
        return jsonify(error_response), 404

    # For other paths, try to serve the UI
    if request.path == '/':
        return serve_ui()
    else:
        return send_from_directory('static', 'index.html')


@app.route('/chat', methods=['POST'])
@validate_request(STARGeneratorRequest)
def chat_with_agent(validated_data: STARGeneratorRequest):
    """
    Process chat requests to the agent.

    Expects a JSON payload with 'role', 'industry', and 'question' fields.
    Returns a structured response with the final STAR answer and iteration history.

    Args:
        validated_data: Pydantic model with validated request data
    """
    # Extract validated fields
    role = validated_data.role
    industry = validated_data.industry
    question_text = validated_data.question

    # Sanitize inputs for LLM to prevent prompt injection
    llm_data = LLMPromptData(
        role=role,
        industry=industry,
        question=question_text
    )

    # Access sanitized values
    app.logger.info(f"Sanitized input: role={llm_data.role}, industry={llm_data.industry}, question={llm_data.question}")

    # Create request details dictionary
    request_details = {
        "role": llm_data.role,
        "industry": llm_data.industry,
        "question": llm_data.question,
        "resume": validated_data.resume,
        "job_description": validated_data.job_description
    }

    # Manage agent session
    agent_session_id = flask_session.get('agent_session_id')
    user_id_for_agent = flask_session.get('user_id_for_agent', 'web_user_' + os.urandom(8).hex())

    try:
        session = get_or_create_session(
            agent_session_id,
            user_id_for_agent,
            request_details,
            flask_session
        )
        agent_session_id = session.id
    except Exception as e:
        app.logger.error(f"Error managing agent session: {e}")
        return jsonify({"error": f"Could not manage agent session: {e}"}), 500

    # Send query to agent
    app.logger.info(f"Using agent session ID: {agent_session_id} for user: {user_id_for_agent}")
    message_to_agent = f"Role = {role}, Industry = {industry}, Question = {question_text}"
    app.logger.info(f"Sending message to agent: {message_to_agent}")

    try:
        # Create user message as Content object
        user_message = Content(
            role="user",
            parts=[Part(text=message_to_agent)]
        )

        # Stream query to agent using Runner
        events = runner.run(
            user_id=user_id_for_agent,
            session_id=agent_session_id,
            new_message=user_message
        )
        
        processed_final_output_dict = None
        raw_agent_text_response = None

        # Process all events
        all_events = list(events)
        app.logger.info(f"Total events received: {len(all_events)}")

        # Add a timestamp to each event for history ordering (without verbose logging)
        for i, event in enumerate(all_events):
            if not hasattr(event, "timestamp"):
                setattr(event, "timestamp", datetime.datetime.now().isoformat())

        # Look for final output in events
        for i, event in enumerate(all_events):
            # Debug all final responses
            if event.is_final_response():
                print(f"[DEBUG] Final response event {i} from {event.author}")

            # Check for FinalOutputRetrieverAgent response
            if event.author == 'FinalOutputRetrieverAgent' and event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(f"[DEBUG] Raw text from FinalOutputRetrieverAgent: {part.text[:500]}...")

                            # Clean up markdown code blocks
                            cleaned_text = _clean_json_string(part.text)

                            try:
                                # Try to fix common JSON issues before parsing
                                if cleaned_text.count('{') > cleaned_text.count('}'):
                                    # Add missing closing braces
                                    cleaned_text += '}' * (cleaned_text.count('{') - cleaned_text.count('}'))
                                elif cleaned_text.count('[') > cleaned_text.count(']'):
                                    # Add missing closing brackets
                                    cleaned_text += ']' * (cleaned_text.count('[') - cleaned_text.count(']'))

                                # Remove trailing commas before closing braces/brackets
                                import re
                                cleaned_text = re.sub(r',\s*}', '}', cleaned_text)
                                cleaned_text = re.sub(r',\s*]', ']', cleaned_text)

                                # Parse the JSON content
                                json_data = json.loads(cleaned_text)
                                print(f"[DEBUG] Parsed JSON keys: {list(json_data.keys())}")

                                # The agent should output the data directly now
                                processed_final_output_dict = json_data
                                print(f"\n[DEBUG] Retrieved output from FinalOutputRetrieverAgent")
                                print(f"[DEBUG] JSON data keys: {list(json_data.keys())}")
                                print(f"[DEBUG] JSON data type: {type(json_data)}")

                                # Add detailed debugging for iteration history
                                if 'history' in processed_final_output_dict:
                                    iterations = processed_final_output_dict.get('history', [])
                                    print(f"[DEBUG] Found {len(iterations)} iterations in 'history'")
                                    if iterations:
                                        print(f"[DEBUG] Type of first history item: {type(iterations[0])}")
                                        if isinstance(iterations[0], dict):
                                            print(f"[DEBUG] First history item keys: {list(iterations[0].keys())}")
                                            print(f"[DEBUG] Full first history item: {iterations[0]}")

                                        # Check ratings in history
                                        for idx, item in enumerate(iterations):
                                            if isinstance(item, dict) and 'critique' in item:
                                                rating = item['critique'].get('rating', 0)
                                                print(f"[DEBUG] Iteration {idx+1} rating: {rating}")
                                    elif 'all_iterations' in processed_final_output_dict:
                                        iterations = processed_final_output_dict.get('all_iterations', [])
                                        print(f"[DEBUG] Found {len(iterations)} iterations in 'all_iterations'")
                                    elif 'interaction_history' in processed_final_output_dict:
                                        iterations = processed_final_output_dict.get('interaction_history', [])
                                        print(f"[DEBUG] Found {len(iterations)} iterations in 'interaction_history'")
                                    else:
                                        print("[DEBUG] WARNING: No iteration history found in output!")
                                        print(f"[DEBUG] Available keys: {list(processed_final_output_dict.keys())}")

                                    raw_agent_text_response = None
                                    break
                                else:
                                    print(f"[DEBUG] No 'retrieved_output' in JSON, using full data")
                                    processed_final_output_dict = json_data
                                    raw_agent_text_response = None
                                    break
                            except json.JSONDecodeError as e:
                                print(f"[DEBUG] JSON decode error: {e}")
                                print(f"[DEBUG] Cleaned text length: {len(cleaned_text)}")
                                print(f"[DEBUG] First 200 chars: {cleaned_text[:200]}")
                                print(f"[DEBUG] Last 200 chars: {cleaned_text[-200:]}")
                                # Save the raw text for later parsing attempts
                                raw_agent_text_response = part.text

            # General final response handling
            elif event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        raw_agent_text_response = part.text

                        try:
                            # Try to parse as JSON
                            cleaned_text = _clean_json_string(part.text)
                            json_data = json.loads(cleaned_text)
                            if 'retrieved_output' in json_data:
                                processed_final_output_dict = json_data['retrieved_output']
                                raw_agent_text_response = None
                                break
                        except json.JSONDecodeError:
                            # Keep as raw text
                            pass


        # Format and return the response
        if processed_final_output_dict:
            # Debug logging
            app.logger.info(f"[DEBUG] Keys in processed_final_output_dict: {list(processed_final_output_dict.keys())}")
            if 'history' in processed_final_output_dict:
                app.logger.info(f"[DEBUG] History length: {len(processed_final_output_dict['history'])}")
                if processed_final_output_dict['history']:
                    app.logger.info(f"[DEBUG] First history item keys: {list(processed_final_output_dict['history'][0].keys())}")

            # Use the simple formatter for a clean, reliable approach
            formatted_response = format_simple_response(processed_final_output_dict)

            # Debug the formatted response
            app.logger.info(f"[DEBUG] Formatted response history length: {len(formatted_response.get('history', []))}")

            # Validate the response against our schema
            try:
                validated_response = validate_response(formatted_response, STARGeneratorResponse)
                app.logger.info(f"[DEBUG] Validated response history length: {len(validated_response.get('history', []))}")
                return jsonify(validated_response)
            except Exception as e:
                app.logger.error(f"Response validation error: {str(e)}")
                # Log the formatted response that failed validation
                app.logger.error(f"[DEBUG] Formatted response that failed validation: {json.dumps(formatted_response, indent=2)}")
                error_response = {
                    "star_answer": None,
                    "feedback": None,
                    "history": [],
                    "metadata": {
                        "status": "ERROR_RESPONSE_VALIDATION",
                        "error_message": f"Response validation failed: {str(e)}"
                    }
                }
                return jsonify(error_response), 500

        elif raw_agent_text_response:
            # Try to parse raw text response as structured output
            cleaned_response = _clean_json_string(raw_agent_text_response)

            try:
                # Try to parse as EnhancedAgentFinalOutput first
                try:
                    parsed_agent_output = EnhancedAgentFinalOutput.model_validate_json(cleaned_response)
                    parsed_dict = parsed_agent_output.model_dump()
                except Exception:
                    # Fall back to legacy format
                    try:
                        parsed_agent_output = AgentFinalOutput.model_validate_json(cleaned_response)
                        parsed_dict = parsed_agent_output.model_dump()
                    except Exception:
                        # Try with raw JSON parsing
                        parsed_dict = json.loads(cleaned_response)

                # Format the response using our simple formatter
                formatted_response = format_simple_response(parsed_dict)

                # Validate the response against our schema
                try:
                    validated_response = validate_response(formatted_response, STARGeneratorResponse)
                    return jsonify(validated_response)
                except Exception as e:
                    app.logger.error(f"Response validation error: {str(e)}")
                    error_response = {
                        "star_answer": None,
                        "feedback": None,
                        "history": [],
                        "metadata": {
                            "status": "ERROR_RESPONSE_VALIDATION",
                            "error_message": f"Response validation failed: {str(e)}"
                        }
                    }
                    return jsonify(error_response), 500
            except Exception as e:
                app.logger.error(f"Error parsing agent response: {e}")

                # Construct an error response in our formatted structure
                request_data = request.get_json()
                error_response = {
                    "star_answer": None,
                    "feedback": None,
                    "history": [],
                    "metadata": {
                        "status": "ERROR_AGENT_PROCESSING",
                        "role": request_data.get("role", ""),
                        "industry": request_data.get("industry", ""),
                        "question": request_data.get("question", ""),
                        "error_message": f"Failed to parse agent response: {str(e)}"
                    }
                }
                return jsonify(error_response), 500
        else:
            # Construct an error response in our formatted structure
            request_data = request.get_json()
            error_response = {
                "star_answer": None,
                "feedback": None,
                "history": [],
                "metadata": {
                    "status": "ERROR_AGENT_PROCESSING",
                    "role": request_data.get("role", ""),
                    "industry": request_data.get("industry", ""),
                    "question": request_data.get("question", ""),
                    "error_message": "Agent did not provide a recognizable final output."
                }
            }
            return jsonify(error_response), 500
    except Exception as e:
        app.logger.error(f"Error during agent query: {e}")
        stack_trace = traceback.format_exc()
        app.logger.error(f"Stack trace: {stack_trace}")

        error_message = f"Agent processing error: {str(e)}"

        # Try to update session state with error information using EventActions if possible
        if session:
            try:
                # Check if we can access EventActions for atomic updates
                event_actions = getattr(session, 'actions', None)

                if event_actions and hasattr(event_actions, 'state_delta'):
                    # Use state_delta for atomic updates
                    event_actions.state_delta = {
                        "final_status": "ERROR_AGENT_PROCESSING",
                        "error_message": error_message,
                        "processing_error": True
                    }
                    app.logger.info("Session state updated with error information via EventActions")
                else:
                    # Fall back to direct state updates
                    session.state["final_status"] = "ERROR_AGENT_PROCESSING"
                    session.state["error_message"] = error_message
                    session.state["processing_error"] = True
                    app.logger.info("Session state updated with error information (direct update)")
            except Exception as state_err:
                app.logger.error(f"Failed to update session state: {state_err}")

        # Construct an error response in our formatted structure
        try:
            request_data = request.get_json()
            error_response = {
                "star_answer": None,
                "feedback": None,
                "history": [],
                "metadata": {
                    "status": "ERROR_AGENT_PROCESSING",
                    "role": request_data.get("role", ""),
                    "industry": request_data.get("industry", ""),
                    "question": request_data.get("question", ""),
                    "error_message": error_message
                }
            }
        except Exception:
            # Fallback if we can't even get the request data
            error_response = {
                "star_answer": None,
                "feedback": None,
                "history": [],
                "metadata": {
                    "status": "ERROR_AGENT_PROCESSING",
                    "error_message": f"Severe error during agent processing: {str(e)}"
                }
            }

        return jsonify(error_response), 500




# Import the clean_json_string function from object_handlers
from .object_handlers import clean_json_string as _clean_json_string

# Note: format_agent_response is imported from response_formatters (line 12)


if __name__ == '__main__':
    # Run the Flask development server
    # Configuration from .flaskenv will be used when running with 'flask run'
    # This direct invocation uses environment variables with fallbacks
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
        host=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.environ.get("FLASK_RUN_PORT", 5004))
    )