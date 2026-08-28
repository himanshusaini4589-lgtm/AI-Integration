from openai import OpenAI

client = OpenAI()

# Store conversation history
conversation = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Answer clearly and concisely."
    }
]

print("AI Assistant started! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Assistant: Goodbye!")
        break

    # Add user message to conversation history
    conversation.append({
        "role": "user",
        "content": user_input
    })

    # Send complete conversation history
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=conversation
    )

    assistant_reply = response.output_text

    # Print response
    print(f"Assistant: {assistant_reply}\n")

    # Add assistant response to conversation history
    conversation.append({
        "role": "assistant",
        "content": assistant_reply
    })