import asyncio
import vertexai

async def main():
    # 1. Initialize your specific GCP Project
    client = vertexai.Client(
        project="agent-1-492509", 
        location="us-central1"
    )

    # 2. Your deployed Fact Checker Resource ID
    agent_resource_name = "projects/611053773155/locations/us-central1/reasoningEngines/6605356236038733824"

    print("🔍 Connecting to your Cloud Fact Checker...")
    remote_app = client.agent_engines.get(name=agent_resource_name)

    # 3. Create a stateful memory session
    print("🧠 Establishing memory session...")
    custom_user_id = "mahtab_local_test"
    remote_session = await remote_app.async_create_session(user_id=custom_user_id)
    current_session_id = remote_session.get("id") if isinstance(remote_session, dict) else remote_session.id
    
    print(f"✅ Session Created: {current_session_id}")
    print("Type a claim to fact-check (or 'exit' to quit).\n")
    print("-" * 50)

    # 4. The Interactive Chat Loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down Fact Checker...")
            break
            
        print("FactChecker: ", end="", flush=True)

        # 5. Stream the request (ADK explicitly expects 'input')
        stream = remote_app.async_stream_query(
            user_id=custom_user_id, 
            session_id=current_session_id, 
            message=user_input  # <--- THE FIX
        )

        # 6. Parse and print the streaming words in real-time
        async for event in stream:
            if isinstance(event, dict) and "content" in event: 
                for part in event["content"].get("parts", []): 
                    if "text" in part: 
                        print(part["text"], end="", flush=True)
            else:
                # Fallback for raw string chunks
                print(event, end="", flush=True)
                
        print("\n" + "-" * 50)

if __name__ == "__main__":
    asyncio.run(main())