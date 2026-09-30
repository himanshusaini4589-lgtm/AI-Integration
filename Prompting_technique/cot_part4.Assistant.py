from dataclasses import dataclass
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
load_dotenv()

@dataclass(frozen=True)
class Provider:
    """One provider to reliably route requests across all inference providers."""

    name: str
    env_var: str
    is_free: bool
    base_url: str | None
    model: str

PROVIDERS = [
    
    Provider(
        "Groq",
        "GROQ_API_KEY",
        True,
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b"
    ),
    Provider(
        "OpenRouter",
        "OPENROUTER_API_KEY",
        True,
        "https://openrouter.ai/api/v1",
        "openai/gpt-sol-latest"
    ),
    Provider(
        "OpenAI",
        "OPENAI_API_KEY",
        True,
        None,
        "gpt-4o-mini"
    ),
]

def select_provider() -> Provider:
    for provider in PROVIDERS:
        if os.getenv(provider.env_var):
            return provider

    expected = ", ".join(p.env_var for p in PROVIDERS)
    raise RuntimeError(
        f"No provider key set, add one of {expected} "
        "to your environment variables"
    )

def build_client(provider: Provider) -> OpenAI:
    api_key = os.getenv(provider.env_var)

    if provider.base_url is None:
        return OpenAI(api_key=api_key)

    return OpenAI(
        api_key=api_key,
        base_url=provider.base_url,
    )

def have_any_key() -> bool:
    return any(os.getenv(p.env_var) for p in PROVIDERS)

print("Found a provider key." if have_any_key() else "No provider key found.")

def llm_reply(prompt: str) -> str:
    provider = select_provider()
    print(f"Using {provider.name} provider")
    client = build_client(provider)
    result = client.chat.completions.create(
        model=provider.model,
        max_tokens=50000,
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )

    return result.choices[0].message.content or ""

SYSTEM_PROMPT = """
You are an expert assistant that solves user queries using multiple short solution steps.

You work in two phases:
- THINKING: provide one brief solution step.
- FINAL_OUTPUT: provide the final answer.

Rules:

1. Reply with EXACTLY ONE JSON object per response.
2. Never output multiple JSON objects in one response.
3. Never output text outside the JSON object.
4. The JSON must have exactly this structure:

{
    "content": "your brief solution step or final answer",
    "step_type": "THINKING"
}

or:

{
    "content": "your final answer",
    "step_type": "FINAL_OUTPUT"
}

5. If more reasoning is needed, output one THINKING step.
6. If the answer is ready, output FINAL_OUTPUT.
"""


def llm_json_reply(messages: list[dict]) -> str:
    provider = select_provider()
    client = build_client(provider)

    kwargs: dict = {
        "model": provider.model,
        "max_tokens": 400,
        "messages": messages,
    }

    result = client.chat.completions.create(**kwargs)
    return result.choices[0].message.content


class CotStep(BaseModel):
    """A step in the CoT process."""

    content: str
    step_type: Literal["THINKING", "FINAL_OUTPUT"] # Thinking -> Plan


def parse_cot_step(raw_reply: str) -> CotStep | str:
    """Parse a raw reply into a CotStep."""
    try:
        parsed = json.loads(raw_reply) # returning a python dict
        return CotStep(**parsed) # converting the python dict to a CotStep object
    except json.JSONDecodeError:
        return f"Invalid JSON: {raw_reply}"


class CotAssistant:
    """An assistant that solves user queries using chain of thought."""

    def __init__(self):
        self.messages: list[dict] = [{"role":"system" , "content":SYSTEM_PROMPT}]
        self.MAX_STEPS = 15

    def run(self, user_query: str) -> str:
        self.messages.append({"role": "user", "content": user_query})

        # print(f"👉🏻 {user_query}\n")

        for _ in range(self.MAX_STEPS):
            raw_reply = llm_json_reply(self.messages) # this will make the llm call and return a string of json format

            self.messages.append({"role": "assistant", "content": raw_reply}) # we append that raw json string into messages list

            parsed = parse_cot_step(raw_reply) # convert the raw json string to a CotStep object

            if isinstance(parsed, CotStep): # check if the parsed is a valid CotStep object
                
                if parsed.step_type == "THINKING":
                    print(f"💡 {parsed.content}\n")
                    continue
                if parsed.step_type == "FINAL_OUTPUT":
                    print(f"🎯 {parsed.content}\n")
                    return parsed.content
            else:
                print(f"❌ {parsed}\n")
                return parsed

        
        # we came out of the loop means we were not able to derive the answer in 15 steps
        return f"Failed to derive the answer in {self.MAX_STEPS} steps."
                




assistant = CotAssistant()

while True:
    user_query = input("👉🏻 Enter your question: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        break
    answer = assistant.run(user_query)
    print(f"🎯 {answer}\n")
    print("--------------------------------")