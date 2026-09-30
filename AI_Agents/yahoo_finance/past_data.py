from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "What is Apple's stock ticker and current price?"}],
        )

print(response.choices[0].message.content)


#this will generate ticker of the time when model was made ,,,,so we need tools to fetch ticker from yahoo finance