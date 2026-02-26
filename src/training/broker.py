"""Broker execution adapter with idempotent orders and retry logic.

Abstracts order submission so the system can switch between paper and live
brokers without changing the execution pipeline.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Immutable order request.  order_id doubles as idempotency key."""

    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    quantity: float
    order_type: str  # "market" | "limit"
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class Fill:
    """Execution report returned by the broker."""

    order_id: str
    fill_id: str
    symbol: str
    side: str
    filled_quantity: float
    fill_price: float
    status: str  # "filled" | "partial" | "rejected" | "error"
    filled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BrokerAdapter(ABC):
    """Abstract broker interface."""

    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        """Submit an order and return a fill report."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Return current status dict for an order."""


class PaperBroker(BrokerAdapter):
    """Simulates fills at current price.  Persists to fills.jsonl.

    Idempotent: returns cached fill on duplicate order_id.
    """

    def __init__(self, paper_dir: str = "data/paper_trading"):
        self.paper_dir = Path(paper_dir)
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.fills_path = self.paper_dir / "fills.jsonl"
        self._fill_cache: dict[str, Fill] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing fills into memory for idempotency checks."""
        if not self.fills_path.exists():
            return
        for line in self.fills_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                fill = Fill(
                    order_id=data["order_id"],
                    fill_id=data["fill_id"],
                    symbol=data["symbol"],
                    side=data["side"],
                    filled_quantity=data["filled_quantity"],
                    fill_price=data["fill_price"],
                    status=data["status"],
                    filled_at=data.get("filled_at", ""),
                    error=data.get("error"),
                    metadata=data.get("metadata", {}),
                )
                self._fill_cache[fill.order_id] = fill
            except (json.JSONDecodeError, KeyError):
                continue

    def _persist_fill(self, fill: Fill) -> None:
        with open(self.fills_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(fill), default=str) + "\n")

    def submit_order(self, order: Order) -> Fill:
        # Idempotency: return cached fill for duplicate order_id
        if order.order_id in self._fill_cache:
            logger.info(
                "PaperBroker: duplicate order_id=%s, returning cached fill",
                order.order_id,
            )
            return self._fill_cache[order.order_id]

        if order.quantity <= 0:
            return Fill(
                order_id=order.order_id,
                fill_id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0.0,
                fill_price=0.0,
                status="error",
                error="Quantity must be positive",
                metadata=order.metadata,
            )

        # Simulate fill at market or limit price with explicit validation.
        if order.order_type == "market":
            current_price = order.metadata.get("current_price")
            if current_price is None:
                return Fill(
                    order_id=order.order_id,
                    fill_id=str(uuid.uuid4()),
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0.0,
                    fill_price=0.0,
                    status="error",
                    error="Market order requires metadata.current_price",
                    metadata=order.metadata,
                )
            fill_price = float(current_price)
        elif order.order_type == "limit":
            if order.limit_price is None:
                return Fill(
                    order_id=order.order_id,
                    fill_id=str(uuid.uuid4()),
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0.0,
                    fill_price=0.0,
                    status="error",
                    error="Limit order requires limit_price",
                    metadata=order.metadata,
                )
            fill_price = float(order.limit_price)
        else:
            return Fill(
                order_id=order.order_id,
                fill_id=str(uuid.uuid4()),
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0.0,
                fill_price=0.0,
                status="error",
                error=f"Unsupported order_type: {order.order_type}",
                metadata=order.metadata,
            )

        fill = Fill(
            order_id=order.order_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            filled_quantity=order.quantity,
            fill_price=fill_price,
            status="filled",
            filled_at=datetime.now(timezone.utc).isoformat(),
            metadata=order.metadata,
        )

        self._fill_cache[order.order_id] = fill
        self._persist_fill(fill)
        logger.info(
            "PaperBroker: filled order %s %s %s qty=%.4f @ %.2f",
            order.order_id[:8],
            order.side,
            order.symbol,
            order.quantity,
            fill.fill_price,
        )
        return fill

    def get_order_status(self, order_id: str) -> dict:
        fill = self._fill_cache.get(order_id)
        if fill is None:
            return {"order_id": order_id, "status": "unknown"}
        return asdict(fill)

    def get_fills(self) -> list[dict]:
        """Return all fills as dicts."""
        return [asdict(f) for f in self._fill_cache.values()]


def submit_with_retry(
    broker: BrokerAdapter,
    order: Order,
    *,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> Fill:
    """Exponential-backoff retry.  Idempotency-safe via order_id."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return broker.submit_order(order)
        except Exception as exc:
            last_error = exc
            wait = backoff_base * (2**attempt)
            logger.warning(
                "Broker submit attempt %d/%d failed: %s  (retry in %.1fs)",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    # All retries exhausted — return error fill
    return Fill(
        order_id=order.order_id,
        fill_id=str(uuid.uuid4()),
        symbol=order.symbol,
        side=order.side,
        filled_quantity=0.0,
        fill_price=0.0,
        status="error",
        filled_at=datetime.now(timezone.utc).isoformat(),
        error=str(last_error),
    )
