# 🔍 Vertex AI Reasoning Engine: Fact-Checker Agent

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4.svg)
![Gemini](https://img.shields.io/badge/Model-Gemini_2.5_Flash-orange.svg)

A serverless, stateful AI agent deployed on **Google Cloud Vertex AI Reasoning Engines**. This project utilizes the Agent Development Kit (ADK) to build a fact-checking assistant that grounds its responses in real-time using the Google Search tool. 

The client interface features an asynchronous, word-by-word streaming pipeline using Server-Sent Events (SSE) and maintains persistent conversation memory across turns.

---

## 🏗️ Architecture

* **Framework:** Google Cloud Agent Development Kit (ADK)
* **Compute:** Vertex AI Reasoning Engines (Serverless containerization)
* **LLM:** `gemini-2.5-flash`
* **Tools:** Native `Google Search` for real-time data grounding
* **Observability:** OpenTelemetry (Traces and prompt capture enabled)

---

## 🚀 Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.10 or higher
* [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install)
* A Google Cloud Project with the **Vertex AI API** enabled.

---

## 🛠️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/gcp-reasoning-engine-factchecker.git
cd gcp-reasoning-engine-factchecker
```

2. Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Authenticate with Google Cloud
```
gcloud auth application-default login
```

4. Environment Variables
Create a .env file in the fact_checker directory.
```
GOOGLE_CLOUD_PROJECT=your-project-id-here
GOOGLE_CLOUD_LOCATION=us-central1

# Observability Flags
GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY=true
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
```

☁️ Deployment
To deploy the agent to Vertex AI, navigate into the fact_checker directory and run the ADK deployment command:
```
cd fact_checker
adk deploy agent_engine .
```

Upon successful deployment, the terminal will output your new Reasoning Engine Resource ID (e.g., projects/123/locations/us-central1/reasoningEngines/456). Save this ID.

💻 Usage (Streaming Client)
To interact with the deployed cloud agent from your local machine, use the included asynchronous streaming client.

Open test_rest.py (or live_client.py).

Update the agent_resource_name variable with your newly deployed Resource ID.

Run the client:
```
python test_rest.py
```

Example Interaction
```
🔍 Connecting to Cloud Agent...
🧠 Creating session...
✅ Session Active: 3753825468520857600

Type a claim to fact-check (or 'exit' to quit).
--------------------------------------------------
User: The Eiffel Tower grows taller in the summer.
Agent: VERDICT: True
REASONING: The iron structure of the Eiffel Tower expands when heated by the summer sun...
SOURCES: [...]
```

📊 Observability & Debugging
Because OpenTelemetry is enabled via the .env file, all interactions are logged to Google Cloud.
To view the agent's thought process, tool execution times, and payload data:

Navigate to the Vertex AI -> Reasoning Engines dashboard in the GCP Console.

Click on the deployed FactChecker agent.

View the Traces tab for a microsecond-level waterfall breakdown of the execution.

For raw container crash logs, run the following in your terminal:
```
gcloud logging read 'resource.type="[aiplatform.googleapis.com/ReasoningEngine](https://aiplatform.googleapis.com/ReasoningEngine)" AND severity>=ERROR' --limit=5
```

