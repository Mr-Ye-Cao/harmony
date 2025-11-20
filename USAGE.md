# Bash Agent with gpt-oss-20b and Harmony Format

This demonstrates how to build an interactive agent that:
1. Receives tasks from users
2. Uses gpt-oss-20b to decide what bash commands to run
3. Parses the LLM's response using the harmony format
4. Executes commands in the terminal
5. Sends results back to the LLM
6. Iterates until the task is complete

## Key Components

### 1. `extract_command()` - Parsing Tool Calls

The core function that extracts bash commands from harmony-formatted messages:

```python
def extract_command(messages: List[Message]) -> Optional[str]:
    """
    Extract bash command from model's generated text.

    Args:
        messages: Parsed messages from harmony encoding

    Returns:
        The bash command string if a tool call was detected, None otherwise
    """
    for msg in messages:
        # Check if message has a recipient (indicates tool call)
        if msg.recipient and "execute_bash" in msg.recipient:
            try:
                if msg.content and len(msg.content) > 0:
                    content_text = msg.content[0].text
                    args = json.loads(content_text)
                    return args.get("command")
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    return None
```

**How it works:**
- Harmony format tool calls have a `recipient` field (e.g., `"functions.execute_bash"`)
- The command arguments are in JSON format in the message content
- We parse the JSON and extract the `command` field

### 2. Harmony Format Example

When the model wants to call a tool, it outputs harmony format like:

```
<|start|>assistant to=functions.execute_bash<|channel|>commentary <|constrain|>json<|message|>{"command": "ls -la"}<|call|>
```

Which parses to:
- **Role**: `assistant`
- **Recipient**: `functions.execute_bash` (indicates tool call)
- **Channel**: `commentary`
- **Content Type**: `<|constrain|>json`
- **Content**: `{"command": "ls -la"}`

### 3. Complete Agent Loop

```python
# 1. Initialize conversation with tool definitions
bash_tool = ToolDescription.new(
    name="execute_bash",
    description="Execute bash commands",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"}
        },
        "required": ["command"]
    }
)

conversation = Conversation.from_messages([
    Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_function_tools([bash_tool])
    ),
    Message.from_role_and_content(Role.USER, "check folders in this directory")
])

# 2. Call model
response = call_model(conversation)

# 3. Parse response for command
# The gpt-oss API may return the command in different ways:
# - Structured tool_calls in the response
# - JSON in the content field: {"command": "ls -la"}
# - Harmony format in the text

content = response["choices"][0]["message"]["content"]

# Try parsing as JSON
try:
    command_json = json.loads(content)
    command = command_json.get("command")
except:
    # Or parse with harmony if it's in that format
    tokens = enc.encode(content, allowed_special="all")
    parsed = enc.parse_messages_from_completion_tokens(tokens)
    command = extract_command(parsed)

# 4. Execute command
output, exit_code = execute_bash_command(command)

# 5. Send result back to LLM
conversation.messages.append(
    Message.from_author_and_content(
        Author.new(Role.TOOL, "functions.execute_bash"),
        f"Output:\n{output}"
    )
)

# 6. Continue loop...
```

## Usage Examples

### Run the interactive agent:

```bash
# Run demo with multiple tasks
python bash_agent.py

# Run with custom task
python bash_agent.py "find all txt files"
python bash_agent.py "count lines in all python files"
python bash_agent.py "show disk usage"
```

### Example Output:

```
🤖 Bash Agent Task: Check what folders are available in this directory
================================================================================

Iteration 1/3
────────────────────────────────────────────────────────────────────────────────
💭 Analysis: User asks to check folders. Should run ls -d */ to list directories.
⚙️  Running: ls -d */
📋 Output (exit 0):
demo/
docs/
javascript/
python/
src/
test-data/
tests/

Iteration 2/3
────────────────────────────────────────────────────────────────────────────────

💬 Final Answer:
Here's a quick list of the sub‑folders in the current directory:

- `demo/`
- `docs/`
- `javascript/`
- `python/`
- `src/`
- `test-data/`
- `tests/`

✅ Agent Complete
```

## Real Execution Results

The agent successfully completed these tasks:

1. **List folders**: Used `ls -d */` and summarized 7 directories
2. **Count Python files**: Used `find . -type f -name "*.py" -print | wc -l` and found 8 files
3. **Show README**: Used `head -n 5 README.md` to display first 5 lines

All commands were:
- ✅ Generated by the LLM
- ✅ Parsed from the response
- ✅ Executed in the actual terminal
- ✅ Results sent back to LLM
- ✅ LLM provided friendly summaries

## Key Points

1. **Harmony Format** is crucial for structured tool calling with gpt-oss models
2. **Tool calls** are identified by the `recipient` field in parsed messages
3. **extract_command()** reliably parses commands from harmony messages
4. **Agent loop** enables true back-and-forth interaction between LLM and terminal
5. **Multiple formats**: The script handles various response formats (JSON, harmony, etc.)

## Files

- `bash_agent.py` - Main interactive agent (recommended)

## Next Steps

You can extend this to:
- Add more tools (file editing, API calls, etc.)
- Implement safety checks for dangerous commands
- Add conversation history persistence
- Create a REPL for continuous interaction
- Handle multi-step complex tasks
