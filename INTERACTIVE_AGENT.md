# Interactive Bash Agent - Working Demo ✅

## What We Built

An **interactive agent** that uses gpt-oss-20b to execute bash commands and complete tasks through an iterative loop.

## The `extract_command` Function

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
        # Tool calls have a recipient field like "functions.execute_bash"
        if msg.recipient and "execute_bash" in msg.recipient:
            try:
                if msg.content and len(msg.content) > 0:
                    content_text = msg.content[0].text
                    # Command is in JSON format
                    args = json.loads(content_text)
                    return args.get("command")
            except (json.JSONDecodeError, AttributeError, KeyError):
                continue
    return None
```

## How It Works

1. **User sends task**: "Check what folders are available in this directory"

2. **LLM generates response** with tool call:
   - In harmony format: `to=functions.execute_bash`
   - Or as JSON: `{"command": "ls -d */"}`

3. **Parser extracts command**:
   - Parse harmony format using `enc.parse_messages_from_completion_tokens()`
   - Use `extract_command()` to get the bash command

4. **Execute command**:
   ```python
   output, exit_code = execute_bash_command("ls -d */")
   ```

5. **Send results back to LLM**:
   ```python
   tool_result = Message.from_author_and_content(
       Author.new(Role.TOOL, "functions.execute_bash"),
       output
   )
   conversation.messages.append(tool_result)
   ```

6. **LLM provides friendly summary**:
   > Here's a quick list of the sub‑folders in the current directory:
   > - `demo/`
   > - `docs/`
   > - `javascript/`
   > - `python/`
   > - `src/`
   > - `test-data/`
   > - `tests/`

## Proven Results

### ✅ Test 1: List Folders
- **User**: "Check what folders are available in this directory"
- **LLM Command**: `ls -d */`
- **Execution**: ✅ Success
- **Result**: Listed 7 directories

### ✅ Test 2: Count Files
- **User**: "Find all Python files and count how many there are"
- **LLM Command**: `find . -type f -name "*.py" -print | wc -l`
- **Execution**: ✅ Success
- **Result**: Found 8 Python files

### ✅ Test 3: Read File
- **User**: "Show me the first 5 lines of README.md"
- **LLM Command**: `head -n 5 README.md`
- **Execution**: ✅ Success
- **Result**: Displayed first 5 lines

### ✅ Test 4: Custom Task
- **User**: "what files are in the python directory"
- **LLM Command**: `ls -la python`
- **Execution**: ✅ Success
- **Result**: Listed contents of python/ directory

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   ```bash
   cp .env.example .env
   # Edit .env and add your actual API key
   ```

3. **Run the agent**:
   ```bash
   # Run with default demo tasks
   python bash_agent.py

   # Run with custom task
   python bash_agent.py "find all markdown files"
   python bash_agent.py "check disk space"
   python bash_agent.py "count total lines in all python files"
   ```

See [SETUP.md](SETUP.md) for detailed configuration instructions.

## Key Features

✅ **True interactivity** - LLM and terminal communicate back and forth
✅ **Command extraction** - Reliably parses commands from harmony format
✅ **Real execution** - Actually runs commands in the terminal
✅ **Error handling** - Handles timeouts and command failures
✅ **Friendly output** - LLM summarizes technical results for users

## The Magic: Harmony Format

The harmony prompt format enables structured tool calling:

```
<|start|>assistant to=functions.execute_bash<|channel|>commentary
<|constrain|>json<|message|>{"command": "ls -la"}<|call|>
```

Parsed into:
- **recipient**: `functions.execute_bash` ← **This is how we know it's a tool call!**
- **content**: `{"command": "ls -la"}` ← **This contains the command**
- **channel**: `commentary`
- **content_type**: `<|constrain|>json`

## Implementation Notes

The gpt-oss-20b API we tested returns commands in the `content` field as JSON:

```json
{
  "role": "assistant",
  "content": "{\"command\":\"ls -la\"}"
}
```

So we handle multiple formats:
1. ✅ Structured API `tool_calls`
2. ✅ JSON in `content` field
3. ✅ Harmony format in text
4. ✅ Bash code blocks

This ensures maximum compatibility!

## What This Proves

🎯 **We can parse LLM output** to extract commands reliably
🎯 **Commands can be executed** in real terminal sessions
🎯 **LLM can iterate** based on command results
🎯 **The harmony format works** for structured tool calling
🎯 **Interactive agents are possible** with gpt-oss-20b

## Next Steps

- ✨ Add more tools (file editing, API calls, calculations)
- 🔒 Add safety checks for dangerous commands
- 💾 Implement conversation persistence
- 🔄 Create a continuous REPL interface
- 🧠 Handle multi-step complex reasoning tasks

---

**Built with**: gpt-oss-20b + OpenAI Harmony Format + Python
