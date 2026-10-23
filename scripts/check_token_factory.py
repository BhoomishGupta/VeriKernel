#!/usr/bin/env python3
"""Nebius Token Factory verification probe for NVIDIA Nemotron models."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# Test both candidate models identified from Token Factory
models_to_test = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
]

for model_id in models_to_test:
    print(f"\n==================================================")
    print(f"Testing Model: {model_id}")
    print(f"==================================================")
    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Write exactly 1 line of C++: // VeriKernel ready",
                }
            ],
            max_tokens=256,
            temperature=0.1,
        )

        choice = res.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", None)

        print(f"Finish Reason : {choice.finish_reason}")
        if reasoning:
            print(f"Reasoning     : {reasoning.strip()[:100]}... (truncated)")
        print(f"Response      : {content.strip()}")
        print(f"Usage         : {res.usage.prompt_tokens} in / {res.usage.completion_tokens} out")

    except Exception as exc:
        print(f"Error testing {model_id}: {exc}")