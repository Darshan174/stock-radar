"""Tests for Phase 10: execution hardening and go-live controls."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.audit import append_audit_event, generate_daily_report, read_audit_events
from training.broker import BrokerAdapter, Order, PaperBroker, submit_with_retry
from training.canary import (
    check_canary_breach,
    enable_canary,
    is_canary_eligible,
    load_canary_state,
    record_canary_trade,
)
from training.paper_trading import PaperPortfolio
from training.pre_trade_risk import check_all_pre_trade_risk


class TestPaperBroker:
    def test_market_order_requires_current_price(self, tmp_path):
        broker = PaperBroker(paper_dir=str(tmp_path))
        order = Order(
            order_id="ord-1",
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            order_type="market",
            metadata={},
        )
        fill = broker.submit_order(order)
        assert fill.status == "error"
        assert "current_price" in (fill.error or "")

    def test_market_order_fills_with_current_price(self, tmp_path):
        broker = PaperBroker(paper_dir=str(tmp_path))
        order = Order(
            order_id="ord-2",
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            order_type="market",
            metadata={"current_price": 187.25},
        )
        fill = broker.submit_order(order)
        assert fill.status == "filled"
        assert fill.fill_price == pytest.approx(187.25, rel=1e-6)

    def test_duplicate_order_id_is_idempotent(self, tmp_path):
        broker = PaperBroker(paper_dir=str(tmp_path))
        order = Order(
            order_id="ord-dup",
            symbol="MSFT",
            side="sell",
            quantity=0.5,
            order_type="market",
            metadata={"current_price": 400.0},
        )
        fill1 = broker.submit_order(order)
        fill2 = broker.submit_order(order)
        assert fill1.fill_id == fill2.fill_id

    def test_submit_with_retry_returns_error_when_adapter_fails(self):
        class FailingBroker(BrokerAdapter):
            def submit_order(self, order: Order):
                raise RuntimeError("boom")

            def get_order_status(self, order_id: str):
                return {"order_id": order_id, "status": "unknown"}

        order = Order(
            order_id="ord-fail",
            symbol="NVDA",
            side="buy",
            quantity=1.0,
            order_type="market",
            metadata={"current_price": 100.0},
        )
        fill = submit_with_retry(
            FailingBroker(),
            order,
            max_retries=2,
            backoff_base=0.0,
        )
        assert fill.status == "error"
        assert fill.filled_quantity == 0.0


class TestPreTradeRisk:
    def test_blocks_oversized_single_position(self):
        result = check_all_pre_trade_risk(
            symbol="AAPL",
            proposed_size=0.30,
            positions={},
            max_single_position=0.20,
        )
        assert result["blocked"] is True
        assert result["checks"]["max_position_size"]["blocked"] is True

    def test_blocks_sector_concentration(self):
        positions = {
            "MSFT": {"position_size": 0.2},
            "GOOGL": {"position_size": 0.15},
        }
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech"}
        result = check_all_pre_trade_risk(
            symbol="AAPL",
            proposed_size=0.10,
            positions=positions,
            sector_map=sector_map,
            max_sector_weight=0.35,
        )
        assert result["blocked"] is True
        assert result["checks"]["sector_concentration"]["blocked"] is True

    def test_blocks_on_daily_loss_cap(self):
        closed_today = [
            {"pnl_pct": -3.0},
            {"pnl_pct": -2.5},
        ]
        result = check_all_pre_trade_risk(
            symbol="AAPL",
            proposed_size=0.05,
            closed_trades_today=closed_today,
            max_daily_loss_pct=5.0,
        )
        assert result["blocked"] is True
        assert result["checks"]["daily_loss_cap"]["blocked"] is True

    def test_blocks_on_total_exposure(self):
        positions = {
            "A": {"position_size": 0.6},
            "B": {"position_size": 0.35},
        }
        result = check_all_pre_trade_risk(
            symbol="C",
            proposed_size=0.10,
            positions=positions,
            max_total_exposure=1.0,
        )
        assert result["blocked"] is True
        assert result["checks"]["total_exposure"]["blocked"] is True

    def test_all_checks_pass_when_within_limits(self):
        result = check_all_pre_trade_risk(
            symbol="AAPL",
            proposed_size=0.05,
            positions={"MSFT": {"position_size": 0.1}},
            closed_trades_today=[{"pnl_pct": 1.0}],
            sector_map={"AAPL": "Tech", "MSFT": "Tech"},
            max_single_position=0.20,
            max_sector_weight=0.35,
            max_daily_loss_pct=5.0,
            max_total_exposure=1.0,
        )
        assert result["blocked"] is False


class TestCanary:
    def test_enable_and_load_canary_state(self, tmp_path):
        state = enable_canary(str(tmp_path))
        loaded = load_canary_state(str(tmp_path))
        assert state["enabled"] is True
        assert loaded["enabled"] is True
        assert loaded["total_trades"] == 0

    def test_canary_symbol_allowlist(self):
        assert is_canary_eligible("AAPL", ["AAPL", "MSFT"]) is True
        assert is_canary_eligible("TSLA", ["AAPL", "MSFT"]) is False
        assert is_canary_eligible("TSLA", []) is True

    def test_record_trade_updates_state(self):
        state = {
            "enabled": True,
            "total_trades": 0,
            "total_pnl_pct": 0.0,
            "breach_count": 0,
            "history": [],
            "disabled_reason": None,
            "updated_at": None,
        }
        state = record_canary_trade(state, 1.25)
        assert state["total_trades"] == 1
        assert state["total_pnl_pct"] == pytest.approx(1.25, abs=1e-6)
        assert len(state["history"]) == 1

    def test_canary_breach_by_trade_limit(self):
        state = {
            "enabled": True,
            "total_trades": 5,
            "total_pnl_pct": 0.0,
            "breach_count": 0,
            "history": [],
            "disabled_reason": None,
            "updated_at": None,
        }
        result = check_canary_breach(state, max_trades=5)
        assert result["breached"] is True


class TestAudit:
    def test_append_and_read_events(self, tmp_path):
        append_audit_event(
            event_type="signal",
            data={"symbol": "AAPL", "signal": "buy"},
            audit_dir=str(tmp_path),
        )
        events = read_audit_events(audit_dir=str(tmp_path))
        assert len(events) == 1
        assert events[0]["event_type"] == "signal"
        assert events[0]["data"]["symbol"] == "AAPL"

    def test_generate_daily_report_counts(self, tmp_path):
        append_audit_event(
            event_type="signal",
            data={"symbol": "AAPL"},
            audit_dir=str(tmp_path),
        )
        append_audit_event(
            event_type="order",
            data={"order_id": "o1"},
            audit_dir=str(tmp_path),
        )
        append_audit_event(
            event_type="fill",
            data={"status": "filled", "pnl_pct": 1.2},
            audit_dir=str(tmp_path),
        )
        append_audit_event(
            event_type="risk_check",
            data={"blocked": True},
            audit_dir=str(tmp_path),
        )
        report = generate_daily_report(audit_dir=str(tmp_path))
        assert report["total_signals"] == 1
        assert report["total_orders"] == 1
        assert report["total_fills"] == 1
        assert report["risk_blocked"] == 1
        assert report["total_pnl_pct"] == pytest.approx(1.2, abs=1e-6)


class TestPaperPortfolioAllowOpen:
    def test_blocked_open_long(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        rec = pp.record_signal(
            symbol="AAPL",
            signal="buy",
            confidence=0.9,
            price=100.0,
            allow_open=False,
        )
        assert rec["action"] == "blocked_open_long"
        assert "AAPL" not in pp.get_open_positions()

    def test_blocked_open_short(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        rec = pp.record_signal(
            symbol="AAPL",
            signal="sell",
            confidence=0.9,
            price=100.0,
            allow_open=False,
        )
        assert rec["action"] == "blocked_open_short"
        assert "AAPL" not in pp.get_open_positions()

    def test_allow_close_even_if_open_blocked(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        pp.record_signal(
            symbol="AAPL",
            signal="buy",
            confidence=0.8,
            price=100.0,
            allow_open=True,
        )
        rec = pp.record_signal(
            symbol="AAPL",
            signal="sell",
            confidence=0.8,
            price=101.0,
            allow_open=False,
        )
        assert rec["action"] == "close_long"
        assert "AAPL" not in pp.get_open_positions()
