# src/agent.py
"""
Finance Purple/White Agent business logic.

Responsibilities:
- Maintain per-question conversation memory.
- Decide whether to answer or call a tool.
- Build prompts for the LLM.
- Parse MCP tool results.
- Prevent redundant tool calls.

This file should NOT know about:
- FastAPI
- A2A server setup
- Agent cards
- MCP SSE transport details
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Protocol, Sequence, Union

from utils.llm_manager import safe_llm_call
from utils.local_llm_wrapper import safe_local_llm_call


class ToolLike(Protocol):
    """Minimal shape needed from an MCP tool object."""

    name: str
    description: str | None


class ToolClient(Protocol):
    """
    Abstract tool client used by the agent.

    The real implementation lives in messenger.py and uses MCP SSE.
    This keeps agent.py transport-agnostic.
    """

    async def list_tools(self) -> Sequence[ToolLike]:
        """Return available tools."""

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return raw MCP result."""


class ConversationMemory:
    """
    Enhanced conversation memory with tool-call tracking.

    Prevents redundant calls to tools that already returned useful data.
    """

    def __init__(self, max_history: int = 10):
        self.history: list[dict[str, Any]] = []
        self.max_history = max_history
        self.tool_call_count: dict[str, int] = {}
        self.successful_tools: set[str] = set()

    def add_tool_call(self, tool: str, params: dict[str, Any], result: Any) -> None:
        """Store a tool call and track whether it produced useful data."""
        self.tool_call_count[tool] = self.tool_call_count.get(tool, 0) + 1

        if self._is_useful_result(result):
            self.successful_tools.add(tool)

        self.history.append(
            {
                "type": "tool_call",
                "tool": tool,
                "params": params,
                "result": result,
                "timestamp": self._get_timestamp(),
                "call_number": self.tool_call_count[tool],
            }
        )

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def add_reasoning(self, thought: str) -> None:
        """Store a reasoning step."""
        if not thought:
            return

        self.history.append(
            {
                "type": "reasoning",
                "thought": thought,
                "timestamp": self._get_timestamp(),
            }
        )

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def should_try_tool(self, tool: str) -> tuple[bool, str]:
        """
        Decide whether a tool should be called again.

        Returns:
            tuple[bool, str]: should_call, reason
        """
        if tool not in self.tool_call_count:
            return True, "First time calling this tool"

        if tool in self.successful_tools:
            return False, f"Already got useful data from {tool}"

        if self.tool_call_count[tool] == 1:
            return True, "Retrying with different parameters"

        if self.tool_call_count[tool] >= 3:
            return False, f"{tool} already tried {self.tool_call_count[tool]} times"

        return True, ""

    def get_tool_usage_summary(self) -> str:
        """Return a compact summary of tool usage."""
        if not self.tool_call_count:
            return "No tools called yet."

        lines = ["Tool Usage Summary:"]
        for tool, count in self.tool_call_count.items():
            status = "✓ Got useful data" if tool in self.successful_tools else "✗ No useful data"
            lines.append(f"  - {tool}: {count} call(s) → {status}")

        return "\n".join(lines)

    def get_summary(self, last_n: int = 3) -> str:
        """Return a compact summary of recent memory."""
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history

        summary: list[str] = []

        for item in recent:
            if item["type"] == "tool_call":
                result_preview = str(item["result"])[:150]
                summary.append(
                    f"[Call #{item['call_number']}] "
                    f"{item['tool']}({item['params']})\n"
                    f"Result: {result_preview}..."
                )
            elif item["type"] == "reasoning":
                summary.append(f"Thought: {item['thought']}")

        return "\n\n".join(summary)

    def clear(self) -> None:
        """Reset memory for a new question."""
        self.history.clear()
        self.tool_call_count.clear()
        self.successful_tools.clear()

    def _is_useful_result(self, result: Any) -> bool:
        """Return True if a tool result looks useful."""
        if isinstance(result, dict):
            if "error" in result:
                return False

            if result.get("timeline") and len(result.get("timeline", [])) > 0:
                return True

            if result.get("data") and len(result.get("data", {})) > 0:
                return True

            if result.get("sections") and len(result.get("sections", {})) > 0:
                return True

            if result.get("company") and not result.get("error"):
                return True

        if isinstance(result, str) and len(result) > 100:
            return True

        return False

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().isoformat()


class FinancePurpleAgent:
    """
    Finance Purple/White Agent.

    This class contains only reasoning/business logic.
    The MCP client is injected through the ToolClient protocol.
    """

    def __init__(self):
        self.name = os.getenv("PURPLE_AGENT_NAME", "finance-purple-agent")

        self.llm_model = os.getenv("LLM_MODEL", "gemini/gemini-2.5-flash-lite")
        self.llm_api_key = os.getenv("LLM_API_KEY")
        self.llm_use_local = bool(int(os.getenv("USE_LOCAL_LLM_WHITE", "1")))

        self.max_iterations = int(os.getenv("WHITE_AGENT_MAX_ITER", "6"))
        self.memory = ConversationMemory(max_history=10)

    async def answer_question(self, question: str, tool_client: ToolClient) -> str:
        """
        Answer one benchmark question using available MCP tools.

        Args:
            question: The finance benchmark question.
            tool_client: A transport-agnostic MCP tool client.

        Returns:
            Final answer string.
        """
        if not question or not question.strip():
            return "ERROR: Missing question"

        self.memory.clear()

        try:
            available_tools = await tool_client.list_tools()
            available_tool_count = len(available_tools)
            max_iterations = min(available_tool_count, self.max_iterations)

            print(f"[PURPLE] Question: {question[:120]}...")
            print(
                f"[PURPLE] Tools ({available_tool_count}): "
                f"{', '.join(tool.name for tool in available_tools)}"
            )
            print(f"[PURPLE] max_iterations={max_iterations}")

            successful_calls = 0
            failed_calls = 0

            for iteration in range(max_iterations):
                print(f"[PURPLE] Iteration {iteration + 1}/{max_iterations}")

                if failed_calls >= 3:
                    print("[PURPLE] Too many failures. Generating final answer from memory.")
                    return await self._generate_final_answer(question)

                prompt = (
                    self._build_initial_prompt(question, available_tools)
                    if iteration == 0
                    else self._build_followup_prompt(question, available_tools)
                )

                decision = await self._get_llm_decision(prompt)

                action = decision.get("action")
                if not action:
                    failed_calls += 1
                    continue

                if action == "answer":
                    reasoning = decision.get("reasoning", "")
                    answer = decision.get("answer", "")
                    self.memory.add_reasoning(reasoning)
                    return str(answer).strip() if answer else "NO_ANSWER_FOUND"

                if action != "tool_call":
                    failed_calls += 1
                    continue

                tool_name = decision.get("tool")
                params = decision.get("params", {}) or {}
                reasoning = decision.get("reasoning", "")

                if not tool_name:
                    failed_calls += 1
                    continue

                should_call, reason = self.memory.should_try_tool(tool_name)
                if not should_call:
                    print(f"[PURPLE] Skipping {tool_name}: {reason}")
                    self.memory.add_reasoning(
                        f"Refused to call {tool_name}: {reason}. "
                        "Need different tool or final answer."
                    )
                    failed_calls += 1
                    continue

                if not any(tool.name == tool_name for tool in available_tools):
                    print(f"[PURPLE] Tool not found: {tool_name}")
                    failed_calls += 1
                    continue

                params = self._normalize_params(params)

                print(f"[PURPLE] Calling tool: {tool_name}")
                print(f"[PURPLE] Params: {params}")
                print(f"[PURPLE] Reasoning: {reasoning}")

                self.memory.add_reasoning(reasoning)

                try:
                    raw_result = await tool_client.call_tool(tool_name, params)
                    result_data = self.extract_text_from_tool_result(raw_result)
                except Exception as exc:
                    result_data = {"error": str(exc)}

                print(f"[PURPLE] Tool result preview: {str(result_data)[:1000]}")

                if isinstance(result_data, dict) and result_data.get("error"):
                    self.memory.add_tool_call(tool_name, params, result_data)
                    self.memory.add_reasoning(
                        f"Tool {tool_name} failed: {result_data.get('error')}"
                    )
                    failed_calls += 1
                    continue

                self.memory.add_tool_call(tool_name, params, result_data)
                successful_calls += 1

                if successful_calls >= 1 and iteration >= 2:
                    print("[PURPLE] Useful data found. Agent should consider answering.")

            print("[PURPLE] Max iterations reached. Generating final answer.")
            return await self._generate_final_answer(question)

        except Exception as exc:
            print(f"[PURPLE] Fatal answer_question error: {exc}")
            return f"ERROR: {exc}"

    async def _get_llm_decision(self, prompt: str) -> dict[str, Any]:
        """Ask the configured LLM for the next action."""
        try:
            if self.llm_use_local:
                response = await safe_local_llm_call(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise JSON-only assistant. Respond ONLY with valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt + "\n\nRespond with ONLY JSON, nothing else.",
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    component="white",
                )
            else:
                response = await safe_llm_call(
                    model=self.llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise JSON-only assistant. Respond ONLY with valid JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt + "\n\nRespond with ONLY JSON, nothing else.",
                        },
                    ],
                    api_key=self.llm_api_key,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

            response_text = response.choices[0].message.content
            print(f"[PURPLE] LLM response preview: {response_text[:300]}")

            return json.loads(response_text)

        except json.JSONDecodeError as exc:
            print(f"[PURPLE] Invalid JSON from LLM: {exc}")
            return {
                "action": "error",
                "reasoning": "LLM returned invalid JSON",
            }
        except Exception as exc:
            print(f"[PURPLE] LLM call failed: {exc}")
            return {
                "action": "error",
                "reasoning": f"LLM call failed: {exc}",
            }

    async def _generate_final_answer(self, question: str) -> str:
        """Generate final answer from memory when loop ends."""
        memory_summary = self.memory.get_summary(last_n=10)

        if not memory_summary:
            return "NO_ANSWER_FOUND"

        prompt = f"""Question: {question}

Research completed:
{memory_summary}

Provide a concise final answer based only on the data obtained.
"""

        try:
            if self.llm_use_local:
                response = await safe_local_llm_call(
                    messages=[
                        {"role": "system", "content": "You are a precise financial assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    component="white",
                )
            else:
                response = await safe_llm_call(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a precise financial assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    api_key=self.llm_api_key,
                    temperature=0.1,
                )

            return response.choices[0].message.content.strip()

        except Exception as exc:
            print(f"[PURPLE] Final answer error: {exc}")
            return "ERROR_GENERATING_ANSWER"

    def _build_initial_prompt(self, question: str, tools: Sequence[ToolLike]) -> str:
        """Build the first reasoning prompt."""
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description or 'No description'}"
            for tool in tools
            if tool.name != "validate_query"
        )

        return f"""PERSONA: You are a precise financial analyst. Respond ONLY with valid JSON.

QUESTION: {question}

AVAILABLE TOOLS:
{tools_desc}

═══════════════════════════════════════════════════════════════════════
CRITICAL DECISION TREE
═══════════════════════════════════════════════════════════════════════

Step 1: Does question involve TIME? For example Q4 FY 2025, latest, recent, 2024.
├─ YES → Call get_today_date_handler FIRST to check if data exists.
└─ NO → Continue to Step 2.

Step 2: Does question need COMPANY TICKER?
├─ Company name is non-standard, abbreviated, or ambiguous.
│  └─ Call get_ticker_symbol_handler first.
│     Then use ticker_symbol in sec_search_handler.
└─ Company name is clear.
   └─ Go directly to the relevant SEC or metrics tool.

Step 3: Which data source?
├─ Need OFFICIAL FILINGS: mergers, board changes, guidance, risk factors.
│  └─ sec_search_handler.
├─ Need QUICK METRICS: revenue, assets, ratios.
│  └─ get_financial_metrics_handler or get_financial_ratios_handler.
└─ Need SPECIFIC DOCUMENT PARSING.
   └─ parse_html_handler + retrieve_info_handler.

RULES:
1. Call only ONE tool per response.
2. If a tool has a `question` parameter, include the original question.
3. If you get ticker from get_ticker_symbol_handler, use it in the next call.
4. For SEC filings, use wide date ranges when appropriate.
5. For future-looking periods, check today's date first.
6. Do not invent data.

RESPOND WITH ONE OF THESE JSON OBJECTS:

{{
  "action": "tool_call",
  "tool": "tool_name",
  "params": {{"param": "value"}},
  "reasoning": "why this tool is needed"
}}

OR:

{{
  "action": "answer",
  "answer": "your concise answer",
  "reasoning": "why this is sufficient"
}}
"""

    def _build_followup_prompt(self, question: str, tools: Sequence[ToolLike]) -> str:
        """Build follow-up prompt with memory and tool-usage awareness."""
        memory_summary = self.memory.get_summary(last_n=3)
        tool_usage = self.memory.get_tool_usage_summary()

        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description or 'No description'}"
            for tool in tools
            if tool.name != "validate_query"
        )

        stuck_warning = ""
        for tool, count in self.memory.tool_call_count.items():
            if count >= 2 and tool in self.memory.successful_tools:
                stuck_warning = f"""
⚠️ CRITICAL WARNING:
You already received useful data from {tool}.
Do NOT call {tool} again.
Either answer using that data or call a DIFFERENT tool.
"""
                break

        return f"""QUESTION: {question}

AVAILABLE TOOLS:
{tools_desc}

═══════════════════════════════════════════════
RECENT WORK
═══════════════════════════════════════════════
{memory_summary}

═══════════════════════════════════════════════
TOOL USAGE
═══════════════════════════════════════════════
{tool_usage}

{stuck_warning}

YOUR OPTIONS:

Option A: Provide final answer:
{{
  "action": "answer",
  "answer": "Based on the data from previous tool calls, the answer is...",
  "reasoning": "I have sufficient information."
}}

Option B: Call a different tool:
{{
  "action": "tool_call",
  "tool": "different_tool_name",
  "params": {{}},
  "reasoning": "Need additional data not provided by previous tools."
}}

RULES:
1. Do NOT call tools marked "✓ Got useful data".
2. Do NOT call the same tool more than twice.
3. If a tool has a `question` parameter, include the original question.
4. Respond with ONLY valid JSON.
"""

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Normalize common parameter aliases from LLM output."""
        normalized = dict(params)

        if "search_term" in normalized and "query" not in normalized:
            normalized["query"] = normalized.pop("search_term")

        if "search_query" in normalized and "query" not in normalized:
            normalized["query"] = normalized.pop("search_query")

        return normalized

    @staticmethod
    def extract_text_from_tool_result(result: Any) -> Union[str, dict[str, str]]:
        """
        Generic parser for MCP tool results.

        Returns:
            str on success.
            {"error": "..."} on failure.
        """
        content_items = getattr(result, "content", None)

        if not content_items:
            return {"error": "Tool returned no content"}

        collected_texts: list[str] = []

        for content in content_items:
            raw_text = getattr(content, "text", None)

            if not raw_text:
                continue

            raw = raw_text.strip()

            if not raw:
                continue

            try:
                data = json.loads(raw)

                if isinstance(data, dict) and data.get("error"):
                    return {"error": str(data["error"])}

                if (
                    isinstance(data, dict)
                    and data.get("extraction_method") == "regex"
                    and data.get("timeline")
                ):
                    if len(data.get("timeline", [])) > 0:
                        collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                        continue

                    return {"error": "REGEX extraction returned empty timeline"}

                if (
                    isinstance(data, dict)
                    and data.get("extraction_method") == "llm_rag"
                    and "answer" in data
                ):
                    answer = data["answer"]
                    if answer and str(answer).strip():
                        collected_texts.append(str(answer).strip())
                        continue

                    return {"error": "LLM+RAG returned empty answer"}

                if isinstance(data, dict):
                    if data:
                        collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                        continue

                if isinstance(data, list):
                    collected_texts.append(json.dumps(data, ensure_ascii=False, indent=2))
                    continue

            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            collected_texts.append(raw)

        final_text = "\n".join(collected_texts).strip()

        if not final_text:
            return {"error": "Tool returned empty or unparseable result"}

        return final_text