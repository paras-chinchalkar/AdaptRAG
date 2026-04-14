import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from src.rag.graph import app
print("Graph compiled successfully!")

# Let's test a simple web search query
try:
    print("Testing routing...")
    inputs = {"question": "Who won the superbowl in 2024?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}': done!")
except Exception as e:
    print(f"Error: {e}")
