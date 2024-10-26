#!/usr/bin/env python3
"""Utilities for working with prompts."""

import json


def replace_in_prompt(
        prompt: list,
        pattern: str,
        content: str,
) -> list:
    """
    Replace occurrences of `pattern` in the messages in `prompt` by `content`.

    prompt is assumed to be a list of messages in the standard OpenAI format.
    (list of dicts with 'role' and 'content' keys).

    Returns a new list of messages after the replacement.
    """
    return [
        {
            **m,
            "content": m["content"].replace(pattern, content)
        }
        for m in prompt
    ]


def load_prompt(prompt_path: str) -> list:
    """Load a prompt from a JSON file."""
    with open(prompt_path, 'r') as f:
        return json.load(f)
