# 🧠 Enterprise Agent Memory: Vertex AI Sessions Integration

\![Python](https://img. চন্দ্রshields.io/badge/Python-3.10+-blue.svg)

This document outlines the architecture, implementation, and troubleshooting protocols for migrating a stateless LLM agent to a stateful, context-aware AI using **Vertex AI Agent Engine Sessions**.

By utilizing Managed Sessions, we offload database provisioning, latency optimization, and token-truncation entirely to Google Cloud infrastructure, while paving the way for long-term user profiling via the **Vertex AI Memory Bank**.

## 🏗️ 1. Architecture Overview

Instead of the user talking directly to the `agent.py` brain, all user inputs are routed through a **Runner**.

1.  The **Runner** intercepts the user's message.
2.  It connects to the **Vertex AI Session Service** to retrieve the user's historical chat thread.
3.  It packages the history and the new message into a strict Google GenAI `Content` object.
4.  It executes the Agent (which can utilize internal tools like Vector Search).
5.  It captures the final response and asynchronously saves the updated state back to the Google Cloud backend.



## ⚙️ 2. Configuration & Prerequisites

You must set strict environment flags to force the generic Google GenAI SDK to route its traffic exclusively through your enterprise Vertex AI infrastructure.

Update your `.env` file with the following variables. *(Note: `AGENT_ENGINE_ID` can be found in the GCP Console under Reasoning Engine -\> your deployed agent).*

```env
# Forces enterprise routing
GOOGLE_GENAI_USE_VERTEXAI="TRUE"

# GCP Project details
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# The specific ID of your deployed Agent Engine
AGENT_ENGINE_ID=1234567890123456789 
```



## 💻 3. Implementation Code

### A. The Agent Update (`agent.py`)

You must explicitly instruct the LLM that it possesses a memory, otherwise it may ignore conversational context (like pronouns).

```python
from google.adk.agents import Agent
# ... existing tool imports ...

root_agent = Agent(
    name="EnterpriseFactChecker",
    model="gemini-2.5-flash",
    instruction="""
    You are an advanced internal fact-checker. 
    
    CRITICAL INSTRUCTION: You now have access to a continuous conversation history. 
    If the user refers to past statements, previous documents, or uses pronouns (like "it", "that", or "they"), you MUST read the session context to understand what they are referring to.
    """,
    tools=[search_internal_knowledge, web_search, check_my_identity] 
)
```

### B. The Memory Manager (`runner.py`)

This script acts as the bridge between your agent logic and Google Cloud's managed session databases.

```python
import os
import asyncio
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.genai import types
from agent import root_agent

load_dotenv()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

APP_NAME = "fact_checker"
USER_ID = "mahtab_engineer"

# 1. Connect to GCP's Managed Memory
session_service = VertexAiSessionService(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    agent_engine_id=os.getenv("AGENT_ENGINE_ID")
)

# 2. Bind the Agent and Memory together
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def main():
    print(f"☁️ Provisioning new cloud session for {USER_ID}...")
    
    # Explicitly provision the session in the cloud backend first
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )
    active_session_id = session.id
    
    print(f"Linked to Cloud Session: {active_session_id}")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        # Format strictly for the GenAI SDK
        content = types.Content(role='user', parts=[types.Part(text=user_input)])
        
        # Stream the async events
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=active_session_id, 
            new_message=content
        ):
            if event.is_final_response():
                print(f"Agent: {event.content.parts[0].text}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

### C. The Audit Tool (`view_sessions.py`)

Because the database is managed by GCP, there is no traditional SQL table UI. Use this script to audit your users' chat histories.

```python
import os
import asyncio
from dotenv import load_dotenv
from google.adk.sessions import VertexAiSessionService

load_dotenv()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

async def inspect_memory():
    session_service = VertexAiSessionService(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        agent_engine_id=os.getenv("AGENT_ENGINE_ID")
    )
    
    APP_NAME = "fact_checker"
    USER_ID = "mahtab_engineer"

    sessions = await session_service.list_sessions(app_name=APP_NAME, user_id=USER_ID)
    
    if not sessions.session_ids:
        print("No sessions found.")
        return

    latest_id = sessions.session_ids[-1] 
    print(f"🧠 Deep Dive into Session: {latest_id}")
    
    session_data = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=latest_id
    )
    print(session_data)

if __name__ == "__main__":
    asyncio.run(inspect_memory())
```

-----

## 🚨 4. ADK Troubleshooting & Common Errors

The Google ADK employs strict parameter validation. If your Runner crashes, consult this matrix:

### Error: `SessionNotFoundError`

  * **Cause:** You attempted to pass a custom, hardcoded string (e.g., `"my-test-session"`) to `runner.run_async()`. GCP tracks sessions via strict UUIDs and blocks unregistered IDs.
  * **Fix:** Use `await session_service.create_session(app_name=..., user_id=...)` first, capture the returned `session.id`, and pass that official ID into the runner.

### Error: `400 INVALID_ARGUMENT` during `get_session`

  * **Cause:** You attempted to pass `None` as the `session_id` into `runner.run_async()`, expecting the GenAI API to auto-generate one. The API rejected the malformed network request.
  * **Fix:** Explicitly provision the session using `create_session` prior to starting the interaction loop.

### Error: `Unexpected keyword argument 'message'` or `'input'`

  * **Cause:** The `runner.run_async()` method has updated its parameter requirements in ADK v1.20+.
  * **Fix:** You must pass the user's text inside a strict GenAI `types.Content` object, and assign it to the parameter `new_message`.
      * *Example:* `new_message=types.Content(role='user', parts=[types.Part(text=user_input)])`

### Error: Missing `agent_engine_id` or `project`

  * **Cause:** When initializing `VertexAiSessionService`, older documentation referenced `project_id`.
  * **Fix:** Ensure your initialization exactly matches: `project=os.getenv("GOOGLE_CLOUD_PROJECT")` and explicitly includes your `agent_engine_id`.

### Harmless Console Warnings

  * **`Warning: there are non-text parts in the response: ['function_call']`**: This is standard ADK behavior. It indicates the LLM paused chatting to execute one of your backend Python tools. Safe to ignore.
  * **`LangChainDeprecationWarning: The class VertexAIEmbeddings was deprecated...`**: Upgrade to the newer library by running `pip install -U langchain-google-genai` and changing your import to `GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")`.