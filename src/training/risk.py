"""Risk and position-sizing utilities for signal predictions."""

from __future__ import annotations

from typing import Any

from training.regime import regime_risk_factor


SIGNAL_TO_DIRECTION = {
    "strong_sell": -1.0,
    "sell": -0.5,
    "hold": 0.0,
    "buy": 0.5,
    "strong_buy": 1.0,
}


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _safe_float(value: Any, default: float) -> float:
    try:
        if value in (None, "", "nan"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_position_size(
    *,
    signal: str,
    confidence: float,
    volatility_pct: float | None = None,
    regime: str | None = None,
    risk_factor: float = 1.0,
    target_volatility_pct: float = 2.0,
    min_confidence: float = 0.35,
    max_abs_position: float = 1.0,
    allow_short: bool = True,
) -> dict[str, float | str]:
    """
    Convert signal + confidence into a risk-aware position.

    Returns:
        Dict with signed position in [-max_abs_position, max_abs_position].
    """
    signal_key = str(signal or "hold").strip().lower()
    direction = SIGNAL_TO_DIRECTION.get(signal_key, 0.0)
    if not allow_short:
        direction = max(0.0, direction)

    conf = _clip(_safe_float(confidence, 0.5), 0.0, 1.0)
    rf = max(0.0, _safe_float(risk_factor, 1.0))
    min_conf = _clip(_safe_float(min_confidence, 0.35), 0.0, 0.95)

    # Confidence below threshold should shrink to zero smoothly.
    if conf <= min_conf:
        confidence_scale = 0.0
    else:
        confidence_scale = (conf - min_conf) / max(1e-9, 1.0 - min_conf)

    regime_scale = regime_risk_factor(regime)

    vol_scale = 1.0
    vol = _safe_float(volatility_pct, 0.0)
    target_vol = max(0.1, _safe_float(target_volatility_pct, 2.0))
    if vol > 0:
        vol_scale = _clip(target_vol / vol, 0.2, 1.0)

    raw_position = direction * confidence_scale * rf * regime_scale * vol_scale
    max_pos = max(0.05, _safe_float(max_abs_position, 1.0))
    position = _clip(raw_position, -max_pos, max_pos)

    return {
        "signal": signal_key,
        "regime": str(regime or "neutral").lower(),
        "position_size": round(float(position), 6),
        "position_size_pct": round(abs(float(position)) * 100.0, 3),
        "confidence": round(conf, 6),
        "confidence_scale": round(float(confidence_scale), 6),
        "regime_scale": round(float(regime_scale), 6),
        "volatility_scale": round(float(vol_scale), 6),
        "risk_factor": round(float(rf), 6),
    }


def calculate_stop_take_profit(
    *,
    signal: str,
    entry_price: float,
    atr: float | None = None,
    atr_multiplier_stop: float = 2.0,
    atr_multiplier_target: float = 3.0,
    max_stop_pct: float = 5.0,
    min_risk_reward: float = 1.5,
) -> dict[str, float | None]:
    """
    ATR-based stop-loss and take-profit.

    For buy/strong_buy: stop below entry, target above.
    For sell/strong_sell: reversed.
    For hold or no ATR: returns None values.

    Returns {stop_loss, take_profit, risk_pct, reward_pct, risk_reward_ratio}.
    """
    sig = str(signal or "hold").strip().lower()
    price = _safe_float(entry_price, 0.0)
    atr_val = _safe_float(atr, 0.0) if atr is not None else 0.0

    if sig == "hold" or price <= 0 or atr_val <= 0:
        return {
            "stop_loss": None,
            "take_profit": None,
            "risk_pct": None,
            "reward_pct": None,
            "risk_reward_ratio": None,
        }

    stop_distance = atr_val * atr_multiplier_stop
    max_stop_distance = price * (max_stop_pct / 100.0)
    stop_distance = min(stop_distance, max_stop_distance)

    target_distance = atr_val * atr_multiplier_target
    # Ensure minimum risk/reward ratio
    if stop_distance > 0 and target_distance / stop_distance < min_risk_reward:
        target_distance = stop_distance * min_risk_reward

    is_long = sig in ("buy", "strong_buy")

    if is_long:
        stop_loss = price - stop_distance
        take_profit = price + target_distance
    else:
        stop_loss = price + stop_distance
        take_profit = price - target_distance

    risk_pct = round((stop_distance / price) * 100.0, 4)
    reward_pct = round((target_distance / price) * 100.0, 4)
    rr_ratio = round(target_distance / stop_distance, 4) if stop_distance > 0 else None

    return {
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "risk_reward_ratio": rr_ratio,
    }


def calculate_per_trade_risk(
    *,
    position_size: float,
    stop_loss_pct: float,
    portfolio_value: float = 100_000.0,
    max_risk_per_trade_pct: float = 2.0,
) -> dict[str, float | bool]:
    """
    Check if position x stop distance exceeds max risk budget.
    Returns {risk_amount, risk_pct, within_limits, adjusted_position_size}.
    """
    pos = abs(_safe_float(position_size, 0.0))
    stop_pct = abs(_safe_float(stop_loss_pct, 0.0))
    pv = max(1.0, _safe_float(portfolio_value, 100_000.0))
    max_risk = max(0.01, _safe_float(max_risk_per_trade_pct, 2.0))

    position_value = pos * pv
    risk_amount = position_value * (stop_pct / 100.0)
    risk_pct = (risk_amount / pv) * 100.0

    within_limits = risk_pct <= max_risk

    if within_limits or stop_pct <= 0:
        adjusted = pos
    else:
        max_risk_amount = pv * (max_risk / 100.0)
        adjusted = max_risk_amount / (pv * (stop_pct / 100.0))

    return {
        "risk_amount": round(risk_amount, 2),
        "risk_pct": round(risk_pct, 4),
        "within_limits": within_limits,
        "adjusted_position_size": round(adjusted, 6),
    }
