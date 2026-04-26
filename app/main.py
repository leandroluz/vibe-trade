from __future__ import annotations

import argparse
from typing import Sequence

from app.analyzer import AnalysisResult, analyze_market
from app.config import load_settings
from app.indicators import add_indicators
from app.mt5_client import MT5Client, MT5ConnectionError


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Copiloto técnico V0 para leitura de candles via MetaTrader 5."
    )
    parser.add_argument("--symbol", default=settings.default_symbol, help="Ativo a analisar.")
    parser.add_argument(
        "--timeframe",
        default=settings.default_timeframe,
        help="Timeframe do MT5, por exemplo: M5, M15, H1, D1.",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=settings.candles_count,
        help="Quantidade de candles para análise.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.candles < 50:
        parser.error("--candles deve ser pelo menos 50 para suportar a EMA 50.")

    settings = load_settings()
    client = MT5Client(host=settings.mt5_host, port=settings.mt5_port)

    try:
        client.initialize()
        candles = client.get_candles(
            symbol=args.symbol,
            timeframe=args.timeframe,
            count=args.candles,
        )
        resolved_symbol = candles.attrs.get("resolved_symbol", args.symbol)
        enriched_candles = add_indicators(candles)
        analysis = analyze_market(resolved_symbol, args.timeframe.upper(), enriched_candles)
    except MT5ConnectionError as exc:
        print(f"Erro de conexão MT5: {exc}")
        return 1
    except Exception as exc:
        print(f"Erro inesperado: {exc}")
        return 1
    finally:
        client.shutdown()

    print(format_analysis(analysis))
    return 0


def format_analysis(analysis: AnalysisResult) -> str:
    return "\n".join(
        [
            "=" * 50,
            "VIBE-TRADE | RESUMO TÉCNICO",
            "=" * 50,
            f"Ativo: {analysis.symbol}",
            f"Timeframe: {analysis.timeframe}",
            f"Candles analisados: {analysis.candles_analyzed}",
            f"Preço atual: {_format_price_value(analysis.current_price)}",
            f"Tendência: {analysis.trend}",
            f"RSI 14: {analysis.rsi:.2f}",
            f"ATR 14: {_format_range_value(analysis.atr, analysis.current_price)}",
            f"Máxima últimos 20 candles: {_format_price_value(analysis.recent_high)}",
            f"Mínima últimos 20 candles: {_format_price_value(analysis.recent_low)}",
            f"Viés: {analysis.bias}",
            "=" * 50,
        ]
    )


def _format_price_value(value: float) -> str:
    decimals = _infer_price_decimals(value)
    return f"{value:.{decimals}f}"


def _format_range_value(value: float, reference_price: float) -> str:
    decimals = max(_infer_price_decimals(reference_price), 4)
    return f"{value:.{decimals}f}"


def _infer_price_decimals(value: float) -> int:
    absolute = abs(value)
    if absolute >= 1000:
        return 2
    if absolute >= 100:
        return 2
    if absolute >= 10:
        return 3
    if absolute >= 1:
        return 5
    if absolute >= 0.1:
        return 5
    return 6


if __name__ == "__main__":
    raise SystemExit(main())
