# Enterprise Runbook: Vertex AI Reasoning Engine (ADK) Fact-Checker

## 1. Architectural Overview
This project deploys a serverless, stateful AI agent to Google Cloud Platform (GCP) using the Agent Development Kit (ADK). 
* **Backend:** Vertex AI Agent Engine (Reasoning Engines).
* **Model:** `gemini-2.5-flash`.
* **Capabilities:** Real-time Google Search grounding.
* **Client Interface:** Asynchronous, word-by-word streaming via the modern `vertexai` Python SDK.
* **Observability:** OpenTelemetry (Traces and Message Content Capture enabled).

## 2. Project Structure
Keep your deployment directory strictly scoped to prevent Vertex AI from containerizing unnecessary files.
```text
fact_checker/
├── .env
├── agent.py
└── requirements.txt
```

## 3. Core Implementation

### A. The Agent (`agent.py`)
This is the core definition. ADK automatically looks for the `root_agent` variable during deployment.

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

# ADK explicitly looks for a variable named 'root_agent'
root_agent = Agent(
    name="FactChecker",
    model="gemini-2.5-flash",
    description="A precise fact-checker that verifies claims.",
    instruction="""
    You are a strict and precise fact-checker. 
    When given a claim, you MUST use the google_search tool to find real-time evidence.
    Evaluate the evidence and respond in the following format:
    
    VERDICT: [True / False / Unverified]
    REASONING: [Provide a brief explanation based on the search results]
    SOURCES: [List the URLs you used to verify the claim]
    """,
    tools=[google_search] 
)
```

### B. Environment Configuration (`.env`)
Store your deployment targets and observability flags here. ADK reads this automatically during deployment.

```env
GOOGLE_CLOUD_PROJECT=your-project-id-here
GOOGLE_CLOUD_LOCATION=us-central1

# Observability Flags (Required for Cloud Console Traces)
GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

## 4. Deployment Process
To deploy or update the agent, ensure your terminal is inside the `fact_checker` directory and run the following command. 

*(Note for Windows Users: Use the backtick `` ` `` for multi-line PowerShell commands, or run it on a single line).*

```powershell
adk deploy agent_engine .
```

**Post-Deployment Step:** Copy the `ReasoningEngine` Resource ID from the terminal output (e.g., `projects/.../reasoningEngines/123456789`). You will need this for your client application.

## 5. The Production Client (`live_client.py`)
This script represents a modern frontend backend. It uses the `vertexai` SDK to establish a stateful memory session and streams the response back asynchronously.

```python
import asyncio
import vertexai

async def main():
    # 1. Initialize Client
    client = vertexai.Client(
        project="your-project-id-here", 
        location="us-central1"
    )

    # 2. Target your Deployed Agent
    # UPDATE THIS WITH YOUR LATEST RESOURCE ID AFTER EVERY DEPLOYMENT
    agent_resource_name = "projects/YOUR_PROJECT_NUMBER/locations/us-central1/reasoningEngines/YOUR_RESOURCE_ID"

    print("🔍 Connecting to Cloud Agent...")
    remote_app = client.agent_engines.get(name=agent_resource_name)

    # 3. Establish Stateful Memory Session
    print("🧠 Creating session...")
    custom_user_id = "prod_user_01"
    remote_session = await remote_app.async_create_session(user_id=custom_user_id)
    
    # Handle SDK dictionary response
    current_session_id = remote_session.get("id") if isinstance(remote_session, dict) else remote_session.id
    
    print(f"✅ Session Active: {current_session_id}\n")
    print("-" * 50)

    # 4. Async Stream Loop
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("Agent: ", end="", flush=True)

        # 5. Execute the Stream
        stream = remote_app.async_stream_query(
            user_id=custom_user_id, 
            session_id=current_session_id, 
            message=user_input  # CRITICAL: ADK streaming expects 'message', not 'input'
        )

        # 6. Parse SSE Chunks
        async for event in stream:
            if isinstance(event, dict) and "content" in event: 
                for part in event["content"].get("parts", []): 
                    if "text" in part: 
                        print(part["text"], end="", flush=True)
            else:
                print(event, end="", flush=True)
                
        print("\n" + "-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Integrations & Workflow Automation (n8n)
Because this Reasoning Engine exposes an API, it is highly recommended to integrate this agent into automated data pipelines. **n8n** is a very relevant addition to this architecture. 

You can create an `n8n` workflow that listens for incoming data (e.g., a Slack message, a new row in Google Sheets, or an email), uses an HTTP Request Node to ping your deployed Agent Engine, and then routes the `VERDICT` to the appropriate channel. This transforms your Python script into a fully automated, event-driven enterprise tool.

---

## 7. Troubleshooting & Post-Mortem Guide

When deploying complex ML systems, errors will happen. Here is the historical log of issues encountered in this architecture and their precise fixes.

### Issue 1: `Unsupported api mode: async` or Missing Methods
* **Symptom:** You receive an `AttributeError` stating the agent has no attribute `query` or `async_stream_query`.
* **Root Cause:** The local Python environment is running an outdated version of the `google-cloud-aiplatform` SDK that cannot deserialize ADK's modern async endpoints.
* **Resolution:** 1. `pip install --upgrade google-cloud-aiplatform`
  2. Completely restart your terminal session to clear the old library from Python's memory.

### Issue 2: The `Session ID` Dictionary Error
* **Symptom:** `AttributeError: 'dict' object has no attribute 'session_id'` when calling `async_create_session`.
* **Root Cause:** In newer versions of the `vertexai` SDK, creating a session returns a raw JSON dictionary instead of a Python Object. 
* **Resolution:** Access the ID using dictionary key notation: `remote_session.get("id")` or `remote_session["session_id"]`.

### Issue 3: The 404 NOT_FOUND Error
* **Symptom:** `google.genai.errors.ClientError: 404 NOT_FOUND. {'message': 'The ReasoningEngine does not exist.'}`
* **Root Cause:** The `AGENT_RESOURCE_ID` hardcoded in your Python client points to a container that was overwritten or deleted. Every time you run `adk deploy`, a *new* ID is generated.
* **Resolution:** Check the terminal output from your latest deployment, or navigate to Vertex AI -> Reasoning Engines in the GCP Console to copy the active ID.

### Issue 4: The Silent Container Crash (InvalidRequestError)
* **Symptom:** The script successfully connects and creates a session, but throws a `400 Bad Request` the moment you send a message. The error details say `Reasoning Engine Execution failed`.
* **Root Cause:** A parameter mismatch in the API call. Standard Vertex endpoints expect `input=user_input`, but ADK's `async_stream_query` specifically requires `message=user_input`.
* **Resolution:** Ensure the parameter matches the ADK signature perfectly.
* **How to debug this in the future:** Run the following command in your terminal to fetch the raw Python traceback directly from the cloud container:
  ```powershell
  gcloud logging read 'resource.type="aiplatform.googleapis.com/ReasoningEngine" AND severity>=ERROR' --limit=1
  ```

### Issue 5: IAM Tool Permissions (The "Sandbox" Crash)
* **Symptom:** The agent works flawlessly locally, but crashes in the cloud specifically when trying to use a tool (like BigQuery or Google Search).
* **Root Cause:** Cloud Run / Reasoning Engine containers execute using a default Compute Engine Service Account. This service account likely lacks the specific IAM roles required to access those external APIs.
* **Resolution:** Navigate to GCP IAM & Admin. Find the service account associated with your Agent Engine, and grant it the necessary roles (e.g., `roles/aiplatform.user` or specific data viewer roles).