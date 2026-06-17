"""
LLM client wrappers for OpenAI and Anthropic APIs with usage tracking.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic
from together import Together

from cs4.config import Config


class UsageTracker:
    """Track API usage across all clients."""
    
    _usage_file = Config.LOGS_DIR / "api_usage.txt"
    
    @classmethod
    def log_usage(cls, provider: str, model: str, tokens: int, metadata: Optional[Dict] = None):
        """Log API usage to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_str = f" | {metadata}" if metadata else ""
        
        with open(cls._usage_file, "a") as f:
            f.write(f"{timestamp} | {provider} | {model} | {tokens}{metadata_str}\n")
    
    @classmethod
    def get_total_usage(cls) -> Dict[str, Any]:
        """Calculate total usage statistics."""
        if not cls._usage_file.exists():
            return {"total_tokens": 0, "by_provider": {}}
        
        total_tokens = 0
        by_provider = {}
        
        with open(cls._usage_file, "r") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) >= 4:
                    provider = parts[1]
                    tokens = int(parts[3])
                    total_tokens += tokens
                    
                    if provider not in by_provider:
                        by_provider[provider] = 0
                    by_provider[provider] += tokens
        
        return {
            "total_tokens": total_tokens,
            "by_provider": by_provider
        }


class OpenAIClient:
    """Wrapper for OpenAI API with usage tracking."""
    
    def __init__(self, api_key: Optional[str] = None, log_usage: bool = True):
        self.api_key = api_key or Config.get_api_key("openai")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.log_usage = log_usage
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion."""
        if model is None:
            model = Config.DEFAULT_MODEL
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        if self.log_usage:
            UsageTracker.log_usage(
                provider="openai",
                model=model,
                tokens=response.usage.total_tokens,
                metadata={"prompt_tokens": response.usage.prompt_tokens,
                         "completion_tokens": response.usage.completion_tokens}
            )
        
        return response
    
    def get_response_text(self, response: Any) -> str:
        """Extract text from response."""
        return response.choices[0].message.content.strip()
    
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        **kwargs
    ) -> str:
        """Simplified chat interface."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = self.chat_completion(
            messages=messages,
            model=model,
            **kwargs
        )
        return self.get_response_text(response)

    # --- Batch API (async, ~50% cheaper) -----------------------------------
    def create_batch(self, requests, endpoint: str = "/v1/chat/completions",
                     completion_window: str = "24h"):
        """Upload a JSONL of chat-completion requests and create a batch.

        `requests` is a list of dicts: {"custom_id": str, "body": {...}} where body
        is a /v1/chat/completions request payload (model, messages, ...).
        Returns the created batch object.
        """
        import json
        lines = []
        for r in requests:
            lines.append(json.dumps({
                "custom_id": r["custom_id"],
                "method": "POST",
                "url": endpoint,
                "body": r["body"],
            }))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        up = self.client.files.create(file=("batch_input.jsonl", data), purpose="batch")
        return self.client.batches.create(
            input_file_id=up.id, endpoint=endpoint, completion_window=completion_window
        )

    def retrieve_batch(self, batch_id: str):
        return self.client.batches.retrieve(batch_id)

    def batch_results(self, batch):
        """Yield parsed result objects (dicts) from a completed batch's output file."""
        import json
        out_id = getattr(batch, "output_file_id", None)
        if not out_id:
            return
        text = self.client.files.content(out_id).text
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


class AnthropicClient:
    """Wrapper for Anthropic API with usage tracking."""
    
    def __init__(self, api_key: Optional[str] = None, log_usage: bool = True):
        self.api_key = api_key or Config.get_api_key("anthropic")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=self.api_key)
        self.log_usage = log_usage
    
    def create_message(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        system: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Any:
        """Create a message."""
        # Build parameters dict
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs
        }
        
        # Only add system if provided
        if system is not None:
            params["system"] = system
        
        response = self.client.messages.create(**params)
        
        if self.log_usage:
            # Anthropic usage is in response.usage
            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            UsageTracker.log_usage(
                provider="anthropic",
                model=model,
                tokens=total_tokens,
                metadata={"input_tokens": response.usage.input_tokens,
                         "output_tokens": response.usage.output_tokens}
            )
        
        return response
    
    def get_response_text(self, response: Any) -> str:
        """Extract text from response."""
        return response.content[0].text
    
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = "claude-3-sonnet-20240229",
        **kwargs
    ) -> str:
        """Simplified chat interface."""
        messages = [{"role": "user", "content": user_message}]
        response = self.create_message(
            messages=messages,
            model=model,
            system=system_prompt,
            **kwargs
        )
        return self.get_response_text(response)

    # --- Message Batches API (async, ~50% cheaper) -------------------------
    def create_batch(self, requests):
        """Submit a list of anthropic batch Request objects."""
        return self.client.messages.batches.create(requests=requests)

    def retrieve_batch(self, batch_id: str):
        return self.client.messages.batches.retrieve(batch_id)

    def batch_results(self, batch_id: str):
        """Iterator over per-request results."""
        return self.client.messages.batches.results(batch_id)


class TogetherAIClient:
    """Wrapper for Together.ai API with usage tracking."""
    
    def __init__(self, api_key: Optional[str] = None, log_usage: bool = True):
        self.api_key = api_key or Config.get_api_key("together")
        if not self.api_key:
            raise ValueError("Together.ai API key not provided")
        
        self.client = Together(api_key=self.api_key)
        self.log_usage = log_usage
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        **kwargs
    ) -> Any:
        """Create a chat completion."""
        if model is None:
            model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        if self.log_usage:
            UsageTracker.log_usage(
                provider="together",
                model=model,
                tokens=response.usage.total_tokens,
                metadata={"prompt_tokens": response.usage.prompt_tokens,
                         "completion_tokens": response.usage.completion_tokens}
            )
        
        return response
    
    def get_response_text(self, response: Any) -> str:
        """Extract text from response."""
        message = response.choices[0].message
        content = message.content.strip() if message.content else ""
        
        # Some models (like Qwen reasoning models) output to 'reasoning' field
        if not content and hasattr(message, 'reasoning') and message.reasoning:
            content = message.reasoning.strip()
        
        return content
    
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        model: str = None,
        **kwargs
    ) -> str:
        """Simplified chat interface."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = self.chat_completion(
            messages=messages,
            model=model,
            **kwargs
        )
        return self.get_response_text(response)


def get_total_usage() -> Dict[str, Any]:
    """Get total API usage statistics."""
    return UsageTracker.get_total_usage()
