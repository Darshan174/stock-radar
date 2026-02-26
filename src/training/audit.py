"""Immutable audit trail and daily trading report.

Audit files are date-partitioned: data/audit/audit_YYYY-MM-DD.jsonl
Append-only — never overwrite or delete entries.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def append_audit_event(
    *,
    event_type: str,
    data: dict,
    audit_dir: str = "data/audit",
) -> dict:
    """Append an audit event.  Returns the full event record.

    event_type: signal | risk_check | order | fill | canary_breach | kill_switch
    """
    d = Path(audit_dir)
    d.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    dt_str = now.strftime("%Y-%m-%d")
    path = d / f"audit_{dt_str}.jsonl"

    event = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": now.isoformat(),
        "data": data,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")

    return event


def read_audit_events(
    dt: str | None = None,
    audit_dir: str = "data/audit",
    event_type: str | None = None,
) -> list[dict]:
    """Read audit events for a given date.

    Args:
        dt: date string YYYY-MM-DD.  Defaults to today.
        audit_dir: audit directory
        event_type: optional filter (e.g. "signal", "fill")
    """
    if dt is None:
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    path = Path(audit_dir) / f"audit_{dt}.jsonl"
    if not path.exists():
        return []

    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event_type is not None and ev.get("event_type") != event_type:
            continue
        events.append(ev)

    return events


def generate_daily_report(
    dt: str | None = None,
    audit_dir: str = "data/audit",
    paper_dir: str = "data/paper_trading",
) -> dict:
    """Aggregate signals, orders, fills, risk blocks, PnL for a day.

    Returns a summary dict suitable for printing or persisting.
    """
    if dt is None:
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    events = read_audit_events(dt=dt, audit_dir=audit_dir)

    signals = [e for e in events if e.get("event_type") == "signal"]
    orders = [e for e in events if e.get("event_type") == "order"]
    fills = [e for e in events if e.get("event_type") == "fill"]
    risk_blocks = [e for e in events if e.get("event_type") == "risk_check"]
    kill_switches = [e for e in events if e.get("event_type") == "kill_switch"]
    canary_breaches = [e for e in events if e.get("event_type") == "canary_breach"]

    # PnL from fills
    total_pnl = 0.0
    for f in fills:
        pnl = f.get("data", {}).get("pnl_pct", 0.0)
        try:
            total_pnl += float(pnl)
        except (TypeError, ValueError):
            pass

    filled_count = sum(
        1 for f in fills if f.get("data", {}).get("status") == "filled"
    )
    rejected_count = sum(
        1 for f in fills if f.get("data", {}).get("status") in ("rejected", "error")
    )
    blocked_count = sum(
        1 for r in risk_blocks if r.get("data", {}).get("blocked")
    )

    return {
        "date": dt,
        "total_signals": len(signals),
        "total_orders": len(orders),
        "total_fills": len(fills),
        "filled_count": filled_count,
        "rejected_count": rejected_count,
        "risk_blocked": blocked_count,
        "kill_switch_events": len(kill_switches),
        "canary_breaches": len(canary_breaches),
        "total_pnl_pct": round(total_pnl, 4),
        "total_events": len(events),
    }


def print_daily_report(report: dict) -> None:
    """Pretty-print a daily report dict to stdout."""
    print(f"\n{'='*60}")
    print(f"  DAILY TRADING REPORT — {report['date']}")
    print(f"{'='*60}")
    print(f"  Signals generated:   {report['total_signals']}")
    print(f"  Orders submitted:    {report['total_orders']}")
    print(f"  Fills received:      {report['total_fills']}")
    print(f"    Filled:            {report['filled_count']}")
    print(f"    Rejected/Error:    {report['rejected_count']}")
    print(f"  Risk-blocked trades: {report['risk_blocked']}")
    print(f"  Kill-switch events:  {report['kill_switch_events']}")
    print(f"  Canary breaches:     {report['canary_breaches']}")
    print(f"  Day PnL (est.):      {report['total_pnl_pct']:+.2f}%")
    print(f"  Total audit events:  {report['total_events']}")
    print()
