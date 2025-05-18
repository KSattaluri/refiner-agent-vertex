# STAR Answer Generator

A Flask web app that uses Google's Vertex AI to generate interview answers in STAR format (Situation, Task, Action, Result). The AI iteratively improves answers until they reach a quality rating of 4.6/5.0.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/KSattaluri/refiner-agent-vertex
   cd refiner-agent-vertex
   ```

2. **Set up Google Cloud credentials**
   - Copy `.env.example` to `.env`
   - Add your Google Cloud project details:
     ```
     GOOGLE_CLOUD_PROJECT=your-project-id
     GOOGLE_CLOUD_LOCATION=us-central1
     ```

3. **Install dependencies**
   ```bash
   poetry install
   ```

4. **Run the server**
   ```bash
   poetry run flask run
   ```

5. **Open the app**
   Navigate to `http://localhost:5004` in your browser

## Project Structure

### `/backend` - Flask Web Server
- **`main.py`** - Flask app with `/chat` endpoint
- **`static/index.html`** - Web UI
- **`validation.py`** - Input/output validation
- **`simple_formatter.py`** - Formats responses for the UI
- **`middleware.py`** - Request validation
- **`object_handlers.py`** - Helper utilities

### `/refiner_agent` - AI Agent System
- **`agent.py`** - Root agent using STAROrchestrator (custom agent class)
- **`orchestrator.py`** - Custom orchestrator that manages the workflow
- **`tools.py`** - Functions for history tracking and state management
- **`schemas.py`** - Data models
- **`subagents/`** - Specialized sub-agents:
  - `input_collector/` - Validates inputs
  - `generator/` - Creates initial STAR answer
  - `critique/` - Rates answer quality (1-5)
  - `refiner/` - Improves answer based on feedback

### `/agent_deploy` - Deployment Scripts
- **`local.py`** - Test agent locally
- **`remote.py`** - Deploy to Vertex AI
- **`cleanup.py`** - Remove deployments

## How It Works

1. User submits role, industry, and interview question
2. Input collector validates the request
3. Generator creates initial STAR answer
4. Critique agent rates the answer
5. If rating < 4.6, refiner improves it
6. Process repeats up to 3 times
7. Final answer and history returned to UI

## Key Design Choice

The STAROrchestrator is a custom agent class that replaces Google's LoopAgent. This was necessary because LoopAgent doesn't have a clear way to exit based on conditions (like achieving a target rating). Our custom orchestrator provides precise control over when to stop refining.

## API Example

```bash
curl -X POST http://localhost:5004/chat \
  -H "Content-Type: application/json" \
  -d '{
    "role": "Software Engineer",
    "industry": "Technology", 
    "question": "Tell me about a time you solved a complex problem."
  }'
```

## Response Format

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
  "history": [...]
}
```