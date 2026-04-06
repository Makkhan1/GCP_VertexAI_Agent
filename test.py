import vertexai
from vertexai.preview import reasoning_engines

# 1. Initialize your connection to your GCP Project
vertexai.init(
    project="agent-1-492509",
    location="us-central1"
)

# 2. Connect to your newly deployed Fact Checker
AGENT_RESOURCE_ID = "projects/611053773155/locations/us-central1/reasoningEngines/1757231237174394880"

print("Connecting to Reasoning Engine...")
live_agent = reasoning_engines.ReasoningEngine(AGENT_RESOURCE_ID)

# 3. Ask it a question!
print("Sending claim to Google Cloud...")
response = live_agent.query(
    input="The Great Wall of China is the only human-made object visible from space."
)

# 4. Print the result
print("\n--- AGENT RESPONSE ---")
print(response)