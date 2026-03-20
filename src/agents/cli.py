"""
Interactive CLI for Stock Radar AI Agents.

Usage:
  python main.py research              # interactive research chat
  python main.py research "Is AAPL oversold?"  # single question
  python main.py trade "Analyze TSLA and trade if signal is strong"
"""

from __future__ import annotations

import sys


def _print_steps(result) -> None:
    """Print agent reasoning steps."""
    if not result.steps:
        return
    print(f"\n{'─'*50}")
    print(f"  {len(result.steps)} tool calls | {result.total_tokens} tokens | {result.total_duration_ms}ms")
    print(f"  Model: {result.model_used}")
    print(f"{'─'*50}")
    for step in result.steps:
        args_str = ", ".join(f"{k}={v!r}" for k, v in (step.tool_args or {}).items())
        print(f"  [{step.step_number}] {step.tool_name}({args_str})  [{step.duration_ms}ms]")


def run_research_cli(question: str | None = None) -> None:
    """Run the research agent — single question or interactive loop."""
    from agents.research_agent import ResearchAgent

    agent = ResearchAgent()

    if question:
        print(f"Researching: {question}\n")
        result = agent.ask(question)
        print(result.answer)
        _print_steps(result)
        return

    # Interactive mode
    print("Stock Radar Research Agent")
    print("Ask any financial question. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("research> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        result = agent.ask(q)
        print(f"\n{result.answer}")
        _print_steps(result)
        print()


def run_trade_cli(instruction: str) -> None:
    """Run the trading agent with a single instruction."""
    from agents.trading_agent import TradingAgent

    agent = TradingAgent()
    print(f"Trading Agent: {instruction}\n")

    result = agent.trade(instruction)
    print(result.answer)
    _print_steps(result)
