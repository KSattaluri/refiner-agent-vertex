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

# Configure logging
import logging
logging.basicConfig(level=logging.WARNING)
app.logger.setLevel(logging.WARNING)

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

    # Create or get session and set initial state
    session = None
    if not agent_session_id:
        try:
            app.logger.info(f"Creating new agent session for user: {user_id_for_agent}")
            # Generate a unique session ID if needed
            new_session_id = str(uuid.uuid4())

            # Add status to request details
            initial_state = request_details.copy()
            initial_state["final_status"] = "IN_PROGRESS"

            # Create session with initial state
            session = session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id_for_agent,
                session_id=new_session_id,
                state=initial_state  # Set initial state with request details
            )
            agent_session_id = session.id
            flask_session['agent_session_id'] = agent_session_id
            flask_session['user_id_for_agent'] = user_id_for_agent

            app.logger.info(f"Created session with ID: {agent_session_id}")
        except Exception as e:
            app.logger.error(f"Error creating agent session: {e}")
            return jsonify({"error": f"Could not create agent session: {e}"}), 500
    else:
        try:
            # Get existing session
            session = session_service.get_session(
                app_name=APP_NAME,
                user_id=user_id_for_agent,
                session_id=agent_session_id
            )

            # Update session state with new request details
            for key, value in request_details.items():
                session.state[key] = value

            # Set default status
            session.state["final_status"] = "IN_PROGRESS"

            app.logger.info(f"Updated existing session {agent_session_id} with new request details")
        except Exception as e:
            app.logger.error(f"Error retrieving session: {e}")
            return jsonify({"error": f"Could not retrieve session: {e}"}), 500

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
                setattr(event, "timestamp", f"2025-05-18T{4+i:02d}:{i%60:02d}:00Z")  # Add timestamps for ordering

        # Look for final output in events
        for event in all_events:
            # Check for FinalOutputRetrieverAgent response
            if event.author == 'FinalOutputRetrieverAgent' and event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Clean up markdown code blocks
                            cleaned_text = _clean_json_string(part.text)

                            try:
                                # Parse the JSON content
                                json_data = json.loads(cleaned_text)
                                if 'retrieved_output' in json_data:
                                    processed_final_output_dict = json_data['retrieved_output']
                                    print(f"\n[DEBUG] Retrieved output from FinalOutputRetrieverAgent")

                                    # Add detailed debugging for iteration history
                                    if 'history' in processed_final_output_dict:
                                        iterations = processed_final_output_dict.get('history', [])
                                        print(f"[DEBUG] Found {len(iterations)} iterations in 'history'")
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
                            except json.JSONDecodeError:
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
            # No need to check format - we always use the simple formatter

            # Use the simple formatter for a clean, reliable approach
            formatted_response = format_simple_response(processed_final_output_dict)

            # Debugging information is now hidden in production mode

            # Validate the response against our schema
            try:
                validated_response = validate_response(formatted_response, STARGeneratorResponse)
                return jsonify(validated_response)
            except Exception as e:
                app.logger.error(f"Response validation error: {str(e)}")
                # Still return the response, but log the error
                return jsonify(formatted_response)

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
                    # Still return the formatted response, but log the error
                    return jsonify(formatted_response)
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