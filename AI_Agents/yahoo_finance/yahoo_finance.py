import warnings

from groq.types.chat import ChatCompletionToolParam
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")


import os
import json

from typing import Callable

import yfinance as yf
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class StockPriceResult(BaseModel):
    """Result of the get_stock_price tool"""
    ticker: str = Field(..., description="The ticker symbol of the stock")
    price: float = Field(..., description="The current price of the stock")
    currency: str = Field(..., description="The currency of the stock price")

class StockPriceError(BaseModel):
    """Error result of the get_stock_price tool"""
    error: str = Field(..., description="The error message")

def get_stock_price(ticker_symbol: str) -> str:
    """Fetch the current price of a stock given it's ticker symbol"""
    try:
        stock = yf.Ticker(ticker_symbol.upper())
        price = stock.fast_info["last_price"]
        currency = stock.fast_info["currency"]
        result = StockPriceResult(
            ticker=ticker_symbol.upper(),
            price=price,
            currency=currency
        )
        return result.model_dump_json()

    except Exception as e:
        print(f"Error fetching stock price for {ticker_symbol}: {e}")
        error = StockPriceError(error=str(e))
        return error.model_dump_json()

class GetStockPriceArgs(BaseModel):
    """Validated arguments for the get_stock_price tool"""
    ticker_symbol: str = Field(..., description="The ticker symbol of the stock to get the price of e.g. AAPL, TSLA, etc.")

class ToolCallInfo(BaseModel):
    """Structured representation of a single tool call result"""
    name: str = Field(..., description="The name of the tool that was called")
    arguments: dict = Field(..., description="The arguments passed to the tool")
    result: str = Field(..., description="The result of the tool call")



TOOL_DEFINITION : list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock given it's ticker symbol",
            "parameters": GetStockPriceArgs.model_json_schema(),
        }
    }
]


SYSTEM_PROMPT = """
You are a helpful stock market assistant. When the user asks about a stock price, 
think step-by-step about what you need to do, then call the get_stock_price tool with 
the correct ticker symbol. After receiving the result, present the information in a clear,
and friendly manner.

IMPORTANT: Always call the get_stock_price tool with the correct ticker symbol. Don't pick up
the price from our old conversations, as the price keeps on changing. Donot assume the price
from previous conversations. Always whenever a price for a stock is asked, we need to call the
relevant tools again.

Always reason out loud before acting so the user can follow your chain of thought.
"""

FEW_SHOT_EXAMPLES = [
    # Example 1: Single stock lookup
    {
        "role": "user",
        "content": "How much is Apple stock right now?"
    },
    {
        "role": "assistant",
        "content": (
            "The user is asking about Apple's stock price. Apple's ticker symbol is AAPL."
            "Let me fetch the current price of Apple stock."
        ),
        "tool_calls": [
            {
                "id": "tool_call_01",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "AAPL"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_01",
        "content": json.dumps({"ticker": "AAPL", "price": 200.3, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Apple (AAPL) is currently trading at $200.30 per share in USD."
            "Keep in mind this is the last traded price and may shift slightly by the time you check again."
        )
    },
    # Example 2: Multi stock comparison
    {
        "role": "user",
        "content": "Compare the stock prices of Google and Microsoft"
    },
    {
        "role": "assistant",
        "content": (
            "The user wants to compare two stocks. Google's parent company "
            "Alphabet trades under the ticker symbol GOOGL, and Microsoft trades "
            "under the ticker symbol MSFT."
            "Let me fetch the current prices of both stocks."
        ),
        "tool_calls": [
            {
                "id": "tool_call_02",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "GOOGL"
                    })
                }
            },
            {
                "id": "tool_call_03",
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "arguments": json.dumps({
                        "ticker_symbol": "MSFT"
                    })
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_02",
        "content": json.dumps({"ticker": "GOOGL", "price": 280.12, "currency": "USD"}),
    },
    {
        "role": "tool",
        "tool_call_id": "tool_call_03",
        "content": json.dumps({"ticker": "MSFT", "price": 235.77, "currency": "USD"}),
    },
    {
        "role": "assistant",
        "content": (
            "Here's the comparison:\n\n"
            "- Alphabet/Google (GOOGL) is trading at $280.12 per share in USD.\n"
            "- Microsoft (MSFT) is trading at $235.77 per share in USD.\n\n"
            "Microsoft's stock is currently cheaper than Google's."
        )
    }
      
]

AVAILABLE_TOOLS: dict[str, Callable[..., str]] = {
    "get_stock_price": get_stock_price,
}

TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "get_stock_price": GetStockPriceArgs,
}

def run_agent(user_query: str) -> None:
    """Single turn agentic loop with chain of thought reasoning"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": user_query},
    ]

    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print(f"{'='*60}\n")


    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=TOOL_DEFINITION,
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        messages.append(assistant_msg)

        # print(assistant_msg, assistant_msg.tool_calls)

        if assistant_msg.content:
            print(f"Assistant (thinking): {assistant_msg.content}\n")

        if choice.finish_reason == "stop":
            break

        if not assistant_msg.tool_calls:
            break

        for tool_call in assistant_msg.tool_calls:
            fn_name = tool_call.function.name
            raw_args = json.loads(tool_call.function.arguments)

            arg_model = TOOL_ARG_MODELS[fn_name]

            if arg_model is None:
                raise ValueError(f"No argument model found for tool: {fn_name}")

            validated_args = arg_model.model_validate(raw_args)
            print(f" [Tool Call] {fn_name}({validated_args.model_dump()})")

            fn = AVAILABLE_TOOLS[fn_name]
            result = fn(**validated_args.model_dump())

            raw_result = json.loads(result)

            if "error" in raw_result:
                parsed_result = StockPriceError.model_validate(raw_result)
                print(f" [Tool Error] {parsed_result.error}")
            else:
                parsed_result = StockPriceResult.model_validate(raw_result)
                print(f" [Tool Result] {parsed_result.model_dump_json()}")

            
            call_info = ToolCallInfo(
                name=fn_name,
                arguments=validated_args.model_dump(),
                result=parsed_result.model_dump_json(),
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": call_info.model_dump_json(),
            })



def main():
    print("Stock price agent (type 'quit' to exit) ")
    print("-" * 45)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        run_agent(query)

if __name__ == "__main__":
    main()