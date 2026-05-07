from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import Settings, load_settings
from app.mt5_client import MT5Client, MT5ConnectionError


def ensure_mt5(settings: Settings | None = None) -> MT5Client:
    active_settings = settings or load_settings()
    client = MT5Client(host=active_settings.mt5_host, port=active_settings.mt5_port)
    client.initialize()
    return client


def get_account_snapshot(client: MT5Client) -> dict[str, Any]:
    mt5 = _get_backend(client)
    account_info = mt5.account_info()
    if account_info is None:
        error_code, error_message = mt5.last_error()
        raise MT5ConnectionError(
            f"Falha ao ler account_info do MT5 ({error_code}): {error_message}"
        )

    account = account_info._asdict()
    return {
        "login": account.get("login"),
        "balance": float(account.get("balance") or 0.0),
        "equity": float(account.get("equity") or 0.0),
        "margin": float(account.get("margin") or 0.0),
        "free_margin": float(account.get("margin_free") or 0.0),
        "margin_level": float(account.get("margin_level") or 0.0),
        "currency": account.get("currency"),
        "server": account.get("server"),
        "company": account.get("company"),
        "name": account.get("name"),
    }


def get_positions_snapshot(client: MT5Client) -> list[dict[str, Any]]:
    mt5 = _get_backend(client)
    positions = mt5.positions_get()
    if positions is None:
        error_code, error_message = mt5.last_error()
        raise MT5ConnectionError(
            f"Falha ao ler posições abertas do MT5 ({error_code}): {error_message}"
        )

    result: list[dict[str, Any]] = []
    for position in positions:
        item = position._asdict()
        result.append(
            {
                "ticket": item.get("ticket"),
                "symbol": item.get("symbol"),
                "type": _position_type_name(mt5, int(item.get("type", -1))),
                "type_code": int(item.get("type", -1)),
                "volume": float(item.get("volume") or 0.0),
                "entry": float(item.get("price_open") or 0.0),
                "sl": float(item.get("sl") or 0.0),
                "tp": float(item.get("tp") or 0.0),
                "price_current": float(item.get("price_current") or 0.0),
                "profit": float(item.get("profit") or 0.0),
                "swap": float(item.get("swap") or 0.0),
                "comment": item.get("comment") or "",
                "time": int(item.get("time") or 0),
            }
        )
    return result


def get_tick(client: MT5Client, symbol: str) -> dict[str, Any]:
    mt5 = _get_backend(client)
    resolved_symbol = client.resolve_symbol(symbol)
    tick = mt5.symbol_info_tick(resolved_symbol)
    if tick is None:
        error_code, error_message = mt5.last_error()
        raise MT5ConnectionError(
            f"Falha ao ler tick de {resolved_symbol} ({error_code}): {error_message}"
        )

    tick_data = tick._asdict()
    return {
        "symbol": resolved_symbol,
        "bid": float(tick_data.get("bid") or 0.0),
        "ask": float(tick_data.get("ask") or 0.0),
        "last": float(tick_data.get("last") or 0.0),
        "spread": _compute_spread(tick_data),
        "time": int(tick_data.get("time") or 0),
    }


def get_rates(client: MT5Client, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    return client.get_candles(symbol=symbol, timeframe=timeframe, count=count)


def get_symbol_info(client: MT5Client, symbol: str) -> Any:
    mt5 = _get_backend(client)
    resolved_symbol = client.resolve_symbol(symbol)
    symbol_info = mt5.symbol_info(resolved_symbol)
    if symbol_info is None:
        error_code, error_message = mt5.last_error()
        raise MT5ConnectionError(
            f"Falha ao ler symbol_info de {resolved_symbol} ({error_code}): {error_message}"
        )
    return symbol_info


def order_calc_profit(
    client: MT5Client,
    *,
    direction: str,
    symbol: str,
    volume: float,
    entry: float,
    target_price: float,
) -> float:
    mt5 = _get_backend(client)
    resolved_symbol = client.resolve_symbol(symbol)
    order_type = _order_type_from_direction(mt5, direction)
    profit = mt5.order_calc_profit(order_type, resolved_symbol, volume, entry, target_price)
    if profit is None:
        error_code, error_message = mt5.last_error()
        raise MT5ConnectionError(
            f"Falha ao calcular profit de {resolved_symbol} ({error_code}): {error_message}"
        )
    return float(profit)


def _get_backend(client: MT5Client) -> Any:
    if not client.initialized or client._mt5 is None:
        raise MT5ConnectionError("Conexão com MT5 não foi inicializada.")
    return client._mt5


def _order_type_from_direction(mt5: Any, direction: str) -> int:
    normalized = direction.strip().lower()
    if normalized == "buy":
        return mt5.ORDER_TYPE_BUY
    if normalized == "sell":
        return mt5.ORDER_TYPE_SELL
    raise MT5ConnectionError(f"Direção inválida para cálculo de risco: {direction}")


def _position_type_name(mt5: Any, type_code: int) -> str:
    if type_code == mt5.POSITION_TYPE_BUY:
        return "buy"
    if type_code == mt5.POSITION_TYPE_SELL:
        return "sell"
    return f"unknown:{type_code}"


def _compute_spread(tick_data: dict[str, Any]) -> float:
    ask = float(tick_data.get("ask") or 0.0)
    bid = float(tick_data.get("bid") or 0.0)
    if ask and bid:
        return ask - bid
    return 0.0
