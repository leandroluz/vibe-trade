from __future__ import annotations

import errno
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


class MT5ConnectionError(Exception):
    """Raised when MetaTrader 5 cannot be initialized or queried."""


@dataclass
class MT5Client:
    host: str = "127.0.0.1"
    port: int = 18812
    initialized: bool = False
    _mt5: Any | None = None
    _backend_name: str | None = None

    def initialize(self) -> None:
        if self.initialized:
            return

        mt5 = self._load_backend()

        try:
            initialized = self._initialize_backend(mt5)
        except OSError as exc:
            raise self._wrap_connection_error(exc) from exc

        if not initialized:
            error_code, error_message = mt5.last_error()
            raise MT5ConnectionError(
                f"Falha ao inicializar MT5 via {self._backend_name} "
                f"({error_code}): {error_message}"
            )

        self._mt5 = mt5
        self.initialized = True

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

        self._mt5 = None
        self.initialized = False

    def get_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if not self.initialized or self._mt5 is None:
            raise MT5ConnectionError("Conexão com MT5 não foi inicializada.")

        resolved_symbol = self.resolve_symbol(symbol)
        timeframe_code = self._parse_timeframe(timeframe)
        rates = self._mt5.copy_rates_from_pos(resolved_symbol, timeframe_code, 0, count)

        if rates is None:
            error_code, error_message = self._mt5.last_error()
            raise MT5ConnectionError(
                f"Falha ao buscar candles de {resolved_symbol} ({error_code}): {error_message}"
            )

        candles = pd.DataFrame(rates)
        if candles.empty:
            raise MT5ConnectionError(
                f"Nenhum candle retornado para {resolved_symbol} no timeframe {timeframe}."
            )

        candles["time"] = pd.to_datetime(candles["time"], unit="s")
        candles.attrs["resolved_symbol"] = resolved_symbol
        return candles

    def resolve_symbol(self, symbol: str) -> str:
        if not self.initialized or self._mt5 is None:
            raise MT5ConnectionError("Conexão com MT5 não foi inicializada.")

        requested = symbol.strip().upper()

        direct_match = self._mt5.symbol_info(requested)
        if direct_match is not None:
            self._select_symbol(direct_match.name)
            return direct_match.name

        symbols = self._mt5.symbols_get()
        if not symbols:
            error_code, error_message = self._mt5.last_error()
            raise MT5ConnectionError(
                "Não foi possível listar os símbolos disponíveis no MT5 "
                f"({error_code}): {error_message}"
            )

        exact_candidates = [item.name for item in symbols if item.name.upper() == requested]
        if len(exact_candidates) == 1:
            self._select_symbol(exact_candidates[0])
            return exact_candidates[0]

        prefix_candidates = [
            item.name
            for item in symbols
            if self._matches_requested_symbol(item.name, requested)
        ]
        if len(prefix_candidates) == 1:
            self._select_symbol(prefix_candidates[0])
            return prefix_candidates[0]

        if not prefix_candidates:
            raise MT5ConnectionError(
                f"Símbolo `{symbol}` não encontrado no broker. "
                "Use o nome exibido no Market Watch do MT5."
            )

        raise MT5ConnectionError(
            f"Símbolo `{symbol}` é ambíguo no broker: {', '.join(sorted(prefix_candidates))}"
        )

    def _load_backend(self) -> Any:
        try:
            import MetaTrader5 as mt5

            self._backend_name = "MetaTrader5"
            return mt5
        except ImportError:
            pass

        try:
            from mt5linux import MetaTrader5 as LinuxMetaTrader5

            self._backend_name = "mt5linux"
            try:
                return LinuxMetaTrader5(host=self.host, port=self.port)
            except OSError as exc:
                raise self._wrap_connection_error(exc) from exc
        except ImportError as exc:
            raise MT5ConnectionError(
                "Nenhum backend MT5 disponível. "
                "No Linux com Wine, instale `mt5linux`; no Windows, use `MetaTrader5`."
            ) from exc

    def _initialize_backend(self, mt5: Any) -> bool:
        if self._backend_name == "mt5linux":
            return bool(mt5.initialize())
        return bool(mt5.initialize())

    def _parse_timeframe(self, timeframe: str) -> int:
        if self._mt5 is None:
            raise MT5ConnectionError("Módulo do MT5 não está carregado.")

        normalized = timeframe.strip().upper()
        timeframe_map = {
            "M1": self._mt5.TIMEFRAME_M1,
            "M2": self._mt5.TIMEFRAME_M2,
            "M3": self._mt5.TIMEFRAME_M3,
            "M4": self._mt5.TIMEFRAME_M4,
            "M5": self._mt5.TIMEFRAME_M5,
            "M10": self._mt5.TIMEFRAME_M10,
            "M12": self._mt5.TIMEFRAME_M12,
            "M15": self._mt5.TIMEFRAME_M15,
            "M20": self._mt5.TIMEFRAME_M20,
            "M30": self._mt5.TIMEFRAME_M30,
            "H1": self._mt5.TIMEFRAME_H1,
            "H2": self._mt5.TIMEFRAME_H2,
            "H3": self._mt5.TIMEFRAME_H3,
            "H4": self._mt5.TIMEFRAME_H4,
            "H6": self._mt5.TIMEFRAME_H6,
            "H8": self._mt5.TIMEFRAME_H8,
            "H12": self._mt5.TIMEFRAME_H12,
            "D1": self._mt5.TIMEFRAME_D1,
            "W1": self._mt5.TIMEFRAME_W1,
            "MN1": self._mt5.TIMEFRAME_MN1,
        }

        try:
            return timeframe_map[normalized]
        except KeyError as exc:
            supported = ", ".join(sorted(timeframe_map))
            raise MT5ConnectionError(
                f"Timeframe inválido: {timeframe}. Use um de: {supported}"
            ) from exc

    def _wrap_connection_error(self, exc: OSError) -> MT5ConnectionError:
        if exc.errno == errno.ECONNREFUSED:
            return MT5ConnectionError(
                "Bridge MT5 indisponível em "
                f"{self.host}:{self.port}. "
                "Inicie o MetaTrader 5 no Wine e execute `python -m mt5linux` "
                "no Python do Windows do mesmo prefixo."
            )

        if exc.errno == errno.EPERM:
            return MT5ConnectionError(
                "Acesso negado ao socket do bridge MT5. "
                f"Verifique permissões de rede local para {self.host}:{self.port}."
            )

        return MT5ConnectionError(
            f"Falha ao conectar ao bridge MT5 em {self.host}:{self.port}: {exc}"
        )

    def _select_symbol(self, symbol: str) -> None:
        if self._mt5 is None:
            raise MT5ConnectionError("Módulo do MT5 não está carregado.")

        if not self._mt5.symbol_select(symbol, True):
            error_code, error_message = self._mt5.last_error()
            raise MT5ConnectionError(
                f"Falha ao ativar símbolo {symbol} no Market Watch "
                f"({error_code}): {error_message}"
            )

    @staticmethod
    def _matches_requested_symbol(available_symbol: str, requested_symbol: str) -> bool:
        normalized_available = available_symbol.upper()
        if normalized_available == requested_symbol:
            return True

        base_symbol = re.split(r"[^A-Z0-9]+", normalized_available, maxsplit=1)[0]
        if base_symbol == requested_symbol:
            return True

        return normalized_available.startswith(requested_symbol)
