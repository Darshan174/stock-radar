"""
ReAct (Reason + Act) loop engine for Stock Radar agents.

This is the core that both Research and Trading agents share.
Uses litellm function calling with any model that supports tool_use
(Gemini 2.5 Pro, GPT-4o, Claude Sonnet, etc.).

The engine:
  1. Sends messages + tool definitions to the LLM
  2. If the LLM returns tool_calls, executes each tool
  3. Appends tool results back to the conversation
  4. Repeats until the LLM returns a final text answer or max_steps
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import litellm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A tool the agent can call."""
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]


@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""
    step_number: int
    thought: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    error: str | None = None
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Final result from an agent run."""
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: int = 0
    model_used: str = ""
    tools_called: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  Engine
# ---------------------------------------------------------------------------

class ReActEngine:
    """
    Shared ReAct loop engine.

    Uses litellm's ``tools`` parameter (OpenAI function-calling format),
    which Gemini 2.5 Pro, GPT-4o, and Claude all support.
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        system_prompt: str,
        model: str | None = None,
        fallback_models: list[str] | None = None,
        max_steps: int = 10,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Model selection
        if model:
            self.model = model
        else:
            try:
                from config import settings
                self.model = getattr(settings, "agent_model", "gemini/gemini-2.5-pro")
            except Exception:
                self.model = "gemini/gemini-2.5-pro"

        if fallback_models:
            self.fallback_models = fallback_models
        else:
            try:
                from config import settings
                self.fallback_models = settings.fallback_models
            except Exception:
                self.fallback_models = []

    # ------------------------------------------------------------------
    #  Tool schema (OpenAI function-calling format)
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Build OpenAI-format tool definitions for litellm."""
        schemas = []
        for t in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return schemas

    # ------------------------------------------------------------------
    #  Execute a single tool
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool.function(**args)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", name, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    #  LLM call with fallback
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> Any:
        """Call the LLM with model fallback chain."""
        models_to_try = [self.model] + [
            m for m in self.fallback_models if m != self.model
        ]
        last_error = None
        for model in models_to_try:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return litellm.completion(**kwargs), model
            except Exception as exc:
                last_error = exc
                logger.warning("Agent LLM %s failed: %s", model, exc)
                continue
        raise RuntimeError(f"All agent models failed. Last error: {last_error}")

    # ------------------------------------------------------------------
    #  Synchronous run
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> AgentResult:
        """Execute the full ReAct loop and return the final result."""
        result = AgentResult(question=user_message)
        start = time.time()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = self._tool_schemas()

        for step_num in range(self.max_steps):
            response, model_used = self._call_llm(messages, tool_schemas)
            result.model_used = model_used

            if hasattr(response, "usage") and response.usage:
                result.total_tokens += getattr(response.usage, "total_tokens", 0)

            msg = response.choices[0].message

            # --- Final answer (no tool calls) ---
            if not getattr(msg, "tool_calls", None):
                result.answer = msg.content or ""
                break

            # --- Process tool calls ---
            # Append the assistant message (with tool_calls) to history
            messages.append(msg.model_dump())

            thought = msg.content or ""

            for tool_call in msg.tool_calls:
                fn = tool_call.function
                tool_name = fn.name
                try:
                    tool_args = json.loads(fn.arguments) if fn.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}

                step_start = time.time()
                tool_result = self._execute_tool(tool_name, tool_args)
                step_duration = int((time.time() - step_start) * 1000)

                step = AgentStep(
                    step_number=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=_truncate_result(tool_result),
                    duration_ms=step_duration,
                )
                result.steps.append(step)
                result.tools_called.append(tool_name)

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, default=str)[:8000],
                })

                # Clear thought after first tool call in a batch
                thought = ""
        else:
            # Max steps exhausted — ask LLM for a final summary without tools
            messages.append({
                "role": "user",
                "content": (
                    "You've reached the maximum number of steps. "
                    "Please provide your best answer based on the information gathered so far."
                ),
            })
            response, model_used = self._call_llm(messages, tools=[])
            result.answer = response.choices[0].message.content or "Max steps reached."

        result.total_duration_ms = int((time.time() - start) * 1000)
        return result

    # ------------------------------------------------------------------
    #  Streaming run (yields SSE-compatible events)
    # ------------------------------------------------------------------

    def run_stream(self, user_message: str) -> Generator[dict[str, Any], None, None]:
        """
        Execute the ReAct loop, yielding events at each step.

        Event types:
          {"type": "thought",      "content": "...", "step": N}
          {"type": "tool_call",    "name": "...", "args": {...}, "step": N}
          {"type": "tool_result",  "name": "...", "result": {...}, "step": N, "duration_ms": N}
          {"type": "answer",       "content": "..."}
          {"type": "error",        "message": "..."}
          {"type": "done",         "steps": N, "tokens": N, "duration_ms": N}
        """
        start = time.time()
        total_tokens = 0

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tool_schemas = self._tool_schemas()
        step_count = 0

        for step_num in range(self.max_steps):
            try:
                response, _ = self._call_llm(messages, tool_schemas)
            except RuntimeError as exc:
                yield {"type": "error", "message": str(exc)}
                return

            if hasattr(response, "usage") and response.usage:
                total_tokens += getattr(response.usage, "total_tokens", 0)

            msg = response.choices[0].message

            # --- Final answer ---
            if not getattr(msg, "tool_calls", None):
                yield {"type": "answer", "content": msg.content or ""}
                break

            # --- Tool calls ---
            messages.append(msg.model_dump())

            if msg.content:
                yield {"type": "thought", "content": msg.content, "step": step_num}

            for tool_call in msg.tool_calls:
                fn = tool_call.function
                tool_name = fn.name
                try:
                    tool_args = json.loads(fn.arguments) if fn.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}

                yield {"type": "tool_call", "name": tool_name, "args": tool_args, "step": step_num}

                step_start = time.time()
                tool_result = self._execute_tool(tool_name, tool_args)
                step_duration = int((time.time() - step_start) * 1000)

                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "result": _truncate_result(tool_result),
                    "step": step_num,
                    "duration_ms": step_duration,
                }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, default=str)[:8000],
                })

            step_count = step_num + 1
        else:
            # Max steps — force a final answer
            messages.append({
                "role": "user",
                "content": "Provide your best answer based on the information gathered so far.",
            })
            try:
                response, _ = self._call_llm(messages, tools=[])
                yield {"type": "answer", "content": response.choices[0].message.content or ""}
            except RuntimeError as exc:
                yield {"type": "error", "message": str(exc)}

        duration_ms = int((time.time() - start) * 1000)
        yield {"type": "done", "steps": step_count, "tokens": total_tokens, "duration_ms": duration_ms}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _truncate_result(result: Any, max_items: int = 5, max_str_len: int = 500) -> Any:
    """Truncate tool results to keep message history manageable."""
    if isinstance(result, dict):
        out = {}
        for k, v in result.items():
            if isinstance(v, list) and len(v) > max_items:
                out[k] = v[:max_items]
                out[f"_{k}_truncated"] = f"{len(v)} total, showing {max_items}"
            elif isinstance(v, str) and len(v) > max_str_len:
                out[k] = v[:max_str_len] + "..."
            else:
                out[k] = v
        return out
    if isinstance(result, list) and len(result) > max_items:
        return result[:max_items]
    return result
