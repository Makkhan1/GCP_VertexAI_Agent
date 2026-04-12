# --- runner.py ---
import os
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService
from google.genai import types
from agent import root_agent

# Load environment variables first
load_dotenv()

# Set the critical global flag required by the ADK docs
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"

print("🔌 Initializing Vertex AI Session Service...")

APP_NAME = "fact_checker"
USER_ID = "mahtab_engineer"

# 1. Connect to GCP's Managed Memory (Now with agent_engine_id!)
session_service = VertexAiSessionService(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
    agent_engine_id=os.getenv("AGENT_ENGINE_ID")
)

# 2. Bind the Agent and the Memory Engine together
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- LOCAL TESTING SCRIPT ---
if __name__ == "__main__":
    import asyncio

    async def main():
        print(f"☁️ Provisioning new cloud session for {USER_ID}...")
        
        # 3. Create session exactly as the docs dictate
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID
        )
        
        active_session_id = session.id
        print(f"[System: Linked to Cloud Session: {active_session_id}]")
        print("\n🧠 Enterprise Stateful Agent Booted!")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
                
            print("Agent: Thinking...")
            
            # Format the input
            content = types.Content(role='user', parts=[types.Part(text=user_input)])
            
            # Stream the events
            async for event in runner.run_async(
                user_id=USER_ID, 
                session_id=active_session_id, 
                new_message=content
            ):
                # Print only the final completed thought
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    print(f"Agent: {final_response}\n")

    asyncio.run(main())