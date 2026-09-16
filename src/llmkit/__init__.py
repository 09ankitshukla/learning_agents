"""llmkit -- the shared model layer for the lessons in this repo.

Intentionally small and boring. It exists so lessons can focus on *agent*
concepts instead of re-solving provider plumbing, and so you can swap a model
running on your CPU for a hosted API without editing a single lesson.

    from llmkit import get_client, system, user
    client = get_client()
    reply = client.chat([system("Be brief."), user("What is an agent?")])
    print(reply.text)
"""

from .config import PRESETS, ConfigError, LLMConfig, load_config
from .factory import LLMClient, get_client
from .terminal import console
from .tools import Execution, Tool, ToolError, ToolRegistry
from .types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
    assistant,
    system,
    tool_result,
    user,
)

__all__ = [
    "PRESETS",
    "ConfigError",
    "Execution",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "Tool",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "Usage",
    "assistant",
    "console",
    "get_client",
    "load_config",
    "system",
    "tool_result",
    "user",
]
