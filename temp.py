
def extract_command_from_text(text: str) -> Optional[str]:
    """
    Extract bash command from model's generated text.

    Supports multiple formats:
    1. Markdown code blocks: ```bash\ncommand\n```
    2. Inline code: `command`
    3. Direct command in text

    Args:
        text: Model's generated text

    Returns:
        Extracted command string, or None if no command found
    """
    # Keep original text for Harmony parsing
    original_text = text

    # Remove leading/trailing whitespace
    text = text.strip()

    if not text:
        return None

    # Priority 0: Use OpenAI Harmony Parser if available
    try:
        # Use global encoding if available to avoid reloading (slow)
        if 'enc' not in globals():
             enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
             globals()['enc'] = enc
        else:
             enc = globals()['enc']

        # Pre-processing: Handle repetition loops (e.g. thousands of <|call|> tokens)
        # The model sometimes gets stuck in a loop appending <|call|> tokens.
        # Harmony might fail or take too long to parse this.
        # We truncate the text if we see excessive <|call|> tokens.
        call_token_count = original_text.count("<|call|>")
        if call_token_count > 10:
            # Find the 10th occurrence and truncate after it
            parts = original_text.split("<|call|>")
            # Reconstruct the first 10 parts + tokens
            truncated_text = "<|call|>".join(parts[:11]) # Keep 10 <|call|> tokens
            logger.warning(f"Detected {call_token_count} <|call|> tokens. Truncating text for parsing.")
            tokens = enc.encode(truncated_text, allowed_special="all")
        else:
            tokens = enc.encode(original_text, allowed_special="all")
        
        # Parse messages with strict=False to handle partial/malformed output
        parsed_messages = enc.parse_messages_from_completion_tokens(tokens, role=Role.ASSISTANT, strict=False)
        
        hdl_detected_in_json = False  # Track if we found HDL in JSON commands
        
        for msg in parsed_messages:
            # Verify role is assistant (optional but good practice)
            role = getattr(msg, 'role', None)
            if role is None and hasattr(msg, 'author'):
                role = msg.author.role
            
            # 1. Direct Command Channels (bash, sh, shell, cmd)
            if msg.channel in ['bash', 'sh', 'shell', 'cmd']:
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content_item in msg.content:
                        if hasattr(content_item, 'text'):
                            command = content_item.text.strip()
                            if command:
                                logger.debug(f"Extracted command from channel '{msg.channel}': {command[:100]}")
                                return command

            # 2. Analysis Channel (JSON commands)
            if msg.channel == 'analysis':
                # Extract content
                if hasattr(msg, 'content') and isinstance(msg.content, list):
                    for content_item in msg.content:
                        if hasattr(content_item, 'text'):
                            text_content = content_item.text
                            # Try to parse as JSON command
                            try:
                                import json
                                data = json.loads(text_content)
                                if "cmd" in data and isinstance(data["cmd"], list):
                                    cmd_list = data["cmd"]
                                    if len(cmd_list) > 0:
                                        command = cmd_list[-1]
                                        # Safety check: Ensure it's not Verilog code wrapped in JSON
                                        if looks_like_hdl_code(command):
                                            logger.warning(f"Ignored HDL code in JSON command: {command[:100]}")
                                            hdl_detected_in_json = True
                                            break  # Stop processing this message
                                        
                                        logger.debug(f"Extracted command from Harmony parser (JSON): {command[:100]}")
                                        return command
                            except json.JSONDecodeError:
                                # If not JSON, check if it's a raw bash command (Model drift handling)
                                # The model sometimes outputs raw bash in analysis channel:
                                # <|channel|>analysis<|message|>bash -lc "..."
                                raw_command = text_content.strip()
                                if looks_like_shell_command(raw_command) and not looks_like_hdl_code(raw_command):
                                     logger.debug(f"Extracted raw command from analysis channel: {raw_command[:100]}")
                                     return raw_command
            
            # 3. Final Channel (Code blocks)
            if msg.channel == 'final':
                 if hasattr(msg, 'content') and isinstance(msg.content, list):
                     for content_item in msg.content:
                            if hasattr(content_item, 'text'):
                                text_content = content_item.text
                                # Look for code blocks
                                code_block_pattern = r'```(?:bash|sh|shell)\s*\n(.*?)\n```'
                                code_blocks = re.findall(code_block_pattern, text_content, re.DOTALL | re.IGNORECASE)
                                if code_blocks:
                                    command = "\n".join(block.strip() for block in code_blocks)
                                    logger.debug(f"Extracted {len(code_blocks)} command blocks from Harmony message: {command[:100]}...")
                                    return command

        # If Harmony parsed successfully but found no command (or found HDL in JSON), return None
        # We do NOT fall back to regex if Harmony worked but found nothing.
        if parsed_messages or hdl_detected_in_json:
             logger.debug("Harmony parsed messages but no command found.")
             return None
            
    except ImportError:
        logger.warning("openai-harmony not installed, falling back to regex parsing")
    except Exception as e:
        logger.debug(f"Harmony parsing failed: {e}")

    # Fallback Regex Parsing (ONLY if Harmony failed or not installed)
    # We keep this minimal and strict.
    
    # Strip internal thought tokens for regex parsing
    text = re.sub(r'<\|channel\|>.*?<\|message\|>', '', text)
    text = re.sub(r'<\|end\|>', '', text)
    text = re.sub(r'<\|start\|>', '', text)
    text = re.sub(r'^\s*assistant\s*', '', text)
    text = text.strip()

    if not text:
        return None

    # Priority 1: Markdown code blocks with bash/sh/shell tags ONLY
    code_block_pattern = r'```(?:bash|sh|shell)\s*\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        command = "\n".join(block.strip() for block in code_blocks)
        logger.debug(f"Extracted {len(code_blocks)} command blocks: {command[:100]}...")
        return command

    # We REMOVED the "Priority 2: Inline code" and "Priority 3: Direct command" fallbacks
    # because they were causing the model to execute random text/Verilog as commands.
    
    logger.debug(f"No command found in text (fallback): {text[:200]}")
    return None

    logger.debug(f"No command found in text: {text[:200]}")
    return None

