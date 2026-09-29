from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversation: list[ChatCompletionMessageParam] = [
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
            model="openai/gpt-oss-120b",
            messages=conversation,
        )
        Assistant_reply = response.choices[0].message.content
        conversation.append({
            "role":"assistant",
            "content": Assistant_reply
            })
            
        print(f"Assistant: {Assistant_reply}\n")
        print(conversation)
    except KeyboardInterrupt:
        print("\n\nAssistant: Goodbye!")
        break

    except Exception as e:
        # Remove the unanswered user message
        if conversation[-1]["role"] == "user":
            conversation.pop()

        print(f"Error occurred: {e}\n")
