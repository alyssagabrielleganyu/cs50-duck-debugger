from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MissingAPIKeyError(Exception):
    pass


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide concise, accurate answers."
)


def _use_mock_mode() -> bool:
    return os.getenv("MOCK_OPENAI", "false").strip().lower() in {"1", "true", "yes"}


def _format_input(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "developer"}:
            role = "user"
        formatted.append({"role": role, "content": content})
    return formatted


def _mock_output(messages: List[Dict[str, str]], system_prompt: str) -> str:
    user_messages = [message.get("content", "") for message in messages if message.get("role") == "user"]
    latest_user_text = user_messages[-1] if user_messages else "Tell me what you're working on."
    teacher_mode = "instructor assistant" in system_prompt.lower()
    if teacher_mode:
        return (
            "Mock teacher response: focus on rubric criteria, feedback structure, and debugging guidance. "
            f"Topic: {latest_user_text}"
        )
    return (
        "Mock student response: let's reason step-by-step and start with a small hint. "
        f"Topic: {latest_user_text}"
    )


def create_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    system_prompt: str | None = None,
    extra_instructions: str | None = None,
) -> str:
    base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if _use_mock_mode():
        return _mock_output(messages, system_prompt=base_prompt)

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise MissingAPIKeyError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    formatted_messages = _format_input(messages)

    # Build system message
    system_message = base_prompt
    if extra_instructions:
        system_message = f"{base_prompt}\n\n{extra_instructions}"

    # Add system message to beginning if not already present
    if formatted_messages and formatted_messages[0].get("role") != "system":
        formatted_messages.insert(0, {"role": "system", "content": system_message})

    response = client.chat.completions.create(
        model=model,
        messages=formatted_messages,
        temperature=temperature,
    )
    
    return response.choices[0].message.content.strip()
