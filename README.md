# STAR Answer Generator - Vertex AI Agent with Flask API

This project is a sophisticated AI-powered system that generates high-quality STAR (Situation, Task, Action, Result) interview answers. It combines Google's Vertex AI Agent Development Kit (ADK) with a Flask web interface, providing both API access and a user-friendly UI.

## Features

- 🎯 **Multi-stage answer generation** using specialized agents
- ✨ **Iterative refinement** to achieve quality ratings of 4.6/5.0 or higher
- 📊 **Detailed history tracking** showing all iteration steps
- 🖥️ **Web UI** for easy interaction
- 🔌 **RESTful API** for programmatic access
- 🧩 **Session management** maintaining context across requests
- 🎨 **Custom orchestration** for efficient processing

## Architecture Overview

```
refiner_agent (STAROrchestrator)
  ├── InitializeHistoryAgent - Sets up tracking structures
  ├── InputCollector - Gathers user inputs (role, industry, question)
  ├── STARGeneratorWithHistory - Creates initial STAR answer
  ├── Refinement Loop (max 3 iterations)
  │   ├── STARCritiqueWithHistory - Evaluates quality (1-5 rating)
  │   └── STARRefinerWithHistory - Improves based on feedback
  └── FinalOutputRetrieverAgent - Formats final response
```

## Components

### 1. Agent System (`refiner_agent/`)

The core AI system built with Google ADK:

- **`agent.py`**: Main orchestrator that coordinates the workflow
- **`tools.py`**: Custom tools for history management and state handling
- **`schemas.py`**: Pydantic models for data validation
- **Subagents**:
  - `input_collector/`: Validates and processes inputs
  - `generator/`: Creates initial STAR answers
  - `critique/`: Evaluates answers with strict criteria
  - `refiner/`: Improves answers based on feedback

### 2. Flask Backend (`backend/`)

RESTful API server providing:

- **`main.py`**: Flask application with `/chat` endpoint
- **`validation.py`**: Request/response validation schemas
- **`simple_formatter.py`**: Response formatting for UI
- **`middleware.py`**: Request/response validation middleware

### 3. Web UI (`backend/static/`)

- Intuitive form for input collection
- Real-time processing feedback
- History display with iteration details
- Expandable sections for critiques and refinements

## Installation

1. **Prerequisites**:
   - Python 3.12+
   - Google Cloud Project with Vertex AI enabled
   - Valid Google Cloud API credentials

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd refiner-agent-vertex
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Configure environment**:
   - Copy `.env.example` to `.env`
   - Update with your Google Cloud credentials:
     ```
     GOOGLE_CLOUD_PROJECT=your-project-id
     GOOGLE_CLOUD_LOCATION=your-location
     GOOGLE_API_KEY=your-api-key
     ```

## Usage

### Running the Flask Server

```bash
poetry run flask run
```

The server will start on `http://localhost:5004`

### Using the Web UI

1. Navigate to `http://localhost:5004`
2. Select role, industry, and question from dropdowns
3. Click "Generate STAR Answer"
4. View the generated answer and iteration history

### API Access

Send POST requests to `/chat`:

```bash
curl -X POST http://localhost:5004/chat \
  -H "Content-Type: application/json" \
  -d '{
    "role": "Software Engineer",
    "industry": "Technology",
    "question": "Tell me about a time you solved a complex problem."
  }'
```

### PowerShell Example with Session Management

```powershell
# Create session for cookie persistence
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession

# Create request body
$body = @{
    role = "Data Scientist"
    industry = "Finance"
    question = "Give an example of a time you had to make a difficult decision."
} | ConvertTo-Json

# Send request
$response = Invoke-WebRequest -Uri http://127.0.0.1:5004/chat `
    -Method POST `
    -Headers @{"Content-Type"="application/json"} `
    -Body $body `
    -WebSession $session

# Display formatted response
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

## Response Format

The API returns structured JSON containing:

```json
{
  "star_answer": {
    "situation": "...",
    "task": "...",
    "action": "...",
    "result": "..."
  },
  "feedback": {
    "rating": 4.7,
    "suggestions": ["..."]
  },
  "history": [
    {
      "iteration": 1,
      "star_answer": {...},
      "critique": {
        "rating": 4.2,
        "suggestions": ["..."]
      }
    }
  ],
  "metadata": {
    "status": "COMPLETED",
    "highest_rating": 4.7,
    "role": "Software Engineer",
    "industry": "Technology",
    "question": "..."
  }
}
```

## Quality Criteria

The system evaluates answers on four dimensions (each 25%):

1. **Structure**: Clear STAR format with balanced sections
2. **Relevance**: Tailored to the role and industry
3. **Specificity**: Concrete examples with metrics
4. **Professional Impact**: Demonstrates value and leadership

Answers are refined until they reach 4.6/5.0 or complete 3 iterations.

## Logging and Debugging

- Flask debug mode can be controlled in `.flaskenv`
- Logging verbosity is managed in `backend/main.py`
- To reduce verbose output, the system suppresses Google ADK logging by default

## Development

### Project Structure

```
refiner-agent-vertex/
├── refiner_agent/          # Core agent system
├── backend/                # Flask API server
├── agent_deploy/           # Deployment scripts
├── docs/                   # Documentation
├── pyproject.toml          # Poetry configuration
├── .env                    # Environment variables
└── .flaskenv              # Flask configuration
```

### Testing the Agent Directly

```bash
# Run the ADK web interface
cd refiner-agent-vertex
poetry run adk web

# Or run programmatically
poetry run python agent_deploy/local.py
```

### Key Dependencies

- `google-cloud-aiplatform[adk,agent-engines]`: Vertex AI SDK
- `Flask`: Web framework
- `pydantic`: Data validation
- `python-dotenv`: Environment management

## Deployment

For production deployment to Vertex AI:

1. Configure `agent_deploy/remote.py` with your project details
2. Run deployment script:
   ```bash
   poetry run python agent_deploy/remote.py
   ```

## Troubleshooting

- **Session issues**: Ensure cookies are enabled or use session management (as shown in PowerShell example)
- **Import errors**: Check that virtual environment is activated
- **Verbose logging**: Configure logging levels in `backend/main.py`
- **Port conflicts**: Change `FLASK_RUN_PORT` in `.flaskenv`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[License information here]

## Acknowledgments

Built using Google's Vertex AI Agent Development Kit and Flask framework.