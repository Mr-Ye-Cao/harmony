#!/usr/bin/env python3
"""
Interactive Bash Agent using gpt-oss-20b with Harmony Format

This demonstrates a complete agent loop:
1. User gives a task
2. LLM decides what bash command to run
3. We parse and execute the command
4. Results go back to LLM
5. LLM provides final answer

Usage:
    python bash_agent.py "check folders in this directory"
    python bash_agent.py "find all python files"
    python bash_agent.py "count lines in README.md"
"""

import sys
import os
import json
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    ToolDescription,
    Author,
)

# Load environment variables from .env file
load_dotenv()


def extract_command(messages: List[Message]) -> Optional[str]:
    """
    Extract bash command from model's generated text.

    Args:
        messages: Parsed messages from harmony encoding

    Returns:
        The bash command string if a tool call was detected, None otherwise
    """
    for msg in messages:
        # Check for recipient indicating tool call
        if msg.recipient and "execute_bash" in msg.recipient:
            try:
                if msg.content and len(msg.content) > 0:
                    content_text = msg.content[0].text
                    args = json.loads(content_text)
                    return args.get("command")
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    return None


def execute_bash_command(command: str, cwd: str = "/home/ye/ml-experiments/harmony") -> Tuple[str, int]:
    """Execute a bash command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        output = result.stdout + result.stderr
        return output.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds", -1
    except Exception as e:
        return f"Error: {e}", -1


def call_model(conversation: Conversation) -> Dict[str, Any]:
    """Call gpt-oss-20b with a conversation."""
    # Get API configuration from environment
    api_key = os.getenv("GPT_OSS_API_KEY")
    api_base = os.getenv("GPT_OSS_API_BASE", "http://127.0.0.1:8000")

    if not api_key:
        raise ValueError(
            "GPT_OSS_API_KEY not found in environment. "
            "Please set it in .env file or environment variables."
        )

    # Convert conversation to API message format
    api_messages = []
    for msg in conversation.messages:
        role = str(msg.author.role.value)
        if msg.content:
            content_parts = []
            for c in msg.content:
                if hasattr(c, 'text'):
                    content_parts.append(c.text)
                elif hasattr(c, 'to_dict'):
                    content_parts.append(json.dumps(c.to_dict()))
            content = " ".join(content_parts) if content_parts else ""
            api_messages.append({"role": role, "content": content})

    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "gpt-oss-20b",
        "messages": api_messages,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def run_bash_agent(user_task: str, max_iterations: int = 5, verbose: bool = True):
    """
    Run the bash agent to complete a task.

    Args:
        user_task: Task description from user
        max_iterations: Max agent loop iterations
        verbose: Print detailed progress

    Returns:
        Final response from LLM
    """
    if verbose:
        print("\n" + "="*80)
        print(f"🤖 Bash Agent Task: {user_task}")
        print("="*80)

    # Load harmony encoding
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Define bash tool
    bash_tool = ToolDescription.new(
        name="execute_bash",
        description="Execute bash commands to check files, directories, run commands, etc.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute (e.g., 'ls -la', 'cat file.txt')",
                }
            },
            "required": ["command"],
        },
    )

    # Initialize conversation
    messages = [
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new()
            .with_model_identity("You are a helpful bash assistant that can execute commands.")
            .with_conversation_start_date("2025-11-19"),
        ),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new()
            .with_instructions(
                "When users ask you to perform file/system operations:\n"
                "1. Call execute_bash with the appropriate command\n"
                "2. Wait for results\n"
                "3. Provide a friendly summary\n\n"
                "Always call the tool - don't just suggest commands."
            )
            .with_function_tools([bash_tool]),
        ),
        Message.from_role_and_content(Role.USER, user_task),
    ]

    conversation = Conversation.from_messages(messages)
    final_response = None

    # Agent loop
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'─'*80}")
            print(f"Iteration {iteration + 1}/{max_iterations}")
            print("─"*80)

        # Call LLM
        try:
            response = call_model(conversation)
        except Exception as e:
            print(f"❌ Error calling model: {e}")
            return None

        # Extract assistant message
        if "choices" not in response or not response["choices"]:
            if verbose:
                print("❌ No response from model")
            return None

        choice = response["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        if verbose and reasoning:
            print(f"💭 Analysis: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

        # Extract command
        command = None

        # Check API tool calls
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if "bash" in tc.get("function", {}).get("name", ""):
                    args_str = tc["function"].get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        command = args.get("command")
                    except:
                        pass

        # Check if content is JSON with command
        if not command and content:
            try:
                content_json = json.loads(content.strip())
                if "command" in content_json:
                    command = content_json["command"]
            except json.JSONDecodeError:
                pass

        # Check for harmony format
        if not command:
            full_text = f"{reasoning}\n{content}".strip()
            if "execute_bash" in full_text:
                try:
                    tokens = enc.encode(full_text, allowed_special="all")
                    parsed = enc.parse_messages_from_completion_tokens(tokens, role=None, strict=False)
                    command = extract_command(parsed)
                except:
                    pass

        # Execute command if found
        if command:
            if verbose:
                print(f"⚙️  Running: {command}")

            output, return_code = execute_bash_command(command)

            if verbose:
                print(f"📋 Output (exit {return_code}):")
                print(output[:500] + ('...' if len(output) > 500 else ''))

            # Add messages to conversation
            conversation.messages.append(
                Message.from_role_and_content(Role.ASSISTANT, content or reasoning)
                .with_channel("commentary")
                .with_recipient("functions.execute_bash")
            )

            tool_result = f"Command: {command}\nExit code: {return_code}\nOutput:\n{output}"
            conversation.messages.append(
                Message.from_author_and_content(
                    Author.new(Role.TOOL, "functions.execute_bash"),
                    tool_result,
                )
                .with_channel("commentary")
                .with_recipient("assistant")
            )

        else:
            # No command - final response
            final_response = content if content else reasoning

            if verbose:
                print(f"\n💬 Final Answer:")
                print(final_response)

            # Add to conversation
            if final_response:
                conversation.messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, final_response)
                    .with_channel("final")
                )

            # Task complete
            if choice.get("finish_reason") == "stop":
                break

    if verbose:
        print("\n" + "="*80)
        print("✅ Agent Complete")
        print("="*80)

    return final_response


if __name__ == "__main__":
    # Get task from command line or use default
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        run_bash_agent(task)
    else:
        # Run demo tasks
        demo_tasks = [
            "Check what folders are available in this directory",
            "Find all Python files and count how many there are",
            "Show me the first 5 lines of README.md",
        ]

        print("\n🤖 Bash Agent Demo - Multiple Tasks")
        print("="*80)

        for i, task in enumerate(demo_tasks, 1):
            run_bash_agent(task, max_iterations=3)

            if i < len(demo_tasks):
                print("\n" * 2)
