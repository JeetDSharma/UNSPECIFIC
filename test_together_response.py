#!/usr/bin/env python3
"""
Test script to debug Together.ai response structure
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from together import Together
import os
from dotenv import load_dotenv

load_dotenv()

client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

# Test with a simple prompt
test_prompt = """Write a short 100-word blog about social media marketing.

Instructions:
- Make it engaging
- Include actionable tips
"""

print("Testing Qwen/Qwen3.5-9B response structure...\n")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": test_prompt}],
)

print("=== Response Object Type ===")
print(type(response))
print()

print("=== Response Attributes ===")
print(dir(response))
print()

print("=== Response Dict ===")
print(response.__dict__)
print()

print("=== Choices ===")
print(f"Number of choices: {len(response.choices)}")
print(f"First choice type: {type(response.choices[0])}")
print()

print("=== Message ===")
print(f"Message type: {type(response.choices[0].message)}")
print(f"Message attributes: {dir(response.choices[0].message)}")
print(f"Message dict: {response.choices[0].message.__dict__}")
print()

print("=== Content ===")
print(f"Content: {response.choices[0].message.content}")
print(f"Content type: {type(response.choices[0].message.content)}")
print(f"Content length: {len(response.choices[0].message.content) if response.choices[0].message.content else 0}")
print()

print("=== Usage ===")
print(f"Usage: {response.usage}")
print(f"Total tokens: {response.usage.total_tokens}")
