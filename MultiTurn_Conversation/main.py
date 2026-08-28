from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conversation = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

print("AI Assistant started! Type 'exit' to stop.")
print("Press Ctrl+C to exit.\n")

while True:
    try:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break

        conversation.append({
            "role": "user",
            "content": user_input
        })

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=conversation
        )

        assistant_reply = response.choices[0].message.content

        print(f"Assistant: {assistant_reply}\n")

        conversation.append({
            "role": "assistant",
            "content": assistant_reply
        })

    except KeyboardInterrupt:
        print("\n\nAssistant: Goodbye!")
        break

    except Exception as e:
        # Remove the unanswered user message
        if conversation[-1]["role"] == "user":
            conversation.pop()

        print(f"Error occurred: {e}\n")