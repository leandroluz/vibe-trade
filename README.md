# vibe-trade

Copiloto de trading V0 para análise técnica no terminal com dados do MetaTrader 5.

Este projeto está preparado para rodar no Linux usando `mt5linux` com o MetaTrader 5 aberto via Wine.

## Escopo da V0

- Conecta ao MetaTrader 5
- Busca candles de um ativo
- Calcula EMA 9, EMA 20, EMA 50, RSI 14 e ATR 14
- Gera um resumo técnico no terminal
- Não envia ordens
- Não usa IA nesta etapa

## Requisitos

- Python 3.10+
- MetaTrader 5 instalado e aberto no Wine
- Conta conectada no terminal do MT5
- Python para Windows dentro do mesmo prefixo Wine do MT5
- Pacote `MetaTrader5` instalado nesse Python do Windows

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Crie seu arquivo de ambiente a partir do exemplo:

```bash
cp .env.example .env
```

## Configuração

Arquivo `.env`:

```env
DEFAULT_SYMBOL=EURUSD
DEFAULT_TIMEFRAME=M5
ANALYSIS_PROFILE=equilibrado
ANALYSIS_LOG_PATH=logs/analysis_history.jsonl
CANDLES_COUNT=300
MT5_HOST=127.0.0.1
MT5_PORT=18812
```

## Preparação do bridge MT5 no Linux

O pacote `MetaTrader5` oficial não instala nativamente no Linux. Neste projeto, o acesso ao terminal é feito com `mt5linux`, que expõe a API do MT5 por um bridge local.

Fluxo esperado:

1. Tenha o MetaTrader 5 aberto no Wine.
2. Instale um Python para Windows dentro do Wine.
3. Nesse Python do Windows, instale o pacote oficial:

```bash
wine path\\to\\python.exe -m pip install MetaTrader5
wine path\\to\\python.exe -m pip install mt5linux
```

4. Inicie o servidor do bridge no ambiente Wine:

```bash
wine path\\to\\python.exe -m mt5linux
```

O padrão do `mt5linux` é escutar em `127.0.0.1:18812`, que já está refletido no `.env.example`.

## Execução

Com os valores padrão do `.env`:

```bash
python -m app.main
```

Por padrão, cada execução pode ser persistida em JSONL para histórico e futura integração com IA:

```bash
python -m app.main --log-file logs/analysis_history.jsonl
```

Informando os parâmetros manualmente:

```bash
python -m app.main --symbol EURUSD --timeframe M5 --candles 300
```

Salvando um snapshot local dos candles atuais para replay posterior:

```bash
python -m app.main --symbol EURUSD --save-data data/eurusd_m5.csv
```

Executando a análise a partir de um CSV salvo, sem depender do MT5:

```bash
python -m app.main --data-file data/eurusd_m5.csv --profile equilibrado
```

Em modo contínuo, reavaliando a cada 60 segundos:

```bash
python -m app.main --symbol EURUSD --profile equilibrado --watch 60
```

No modo `--watch`, a tela é atualizada a cada ciclo e o terminal destaca mudanças de setup entre uma leitura e outra.
Com `--data-file`, o `--watch` faz replay candle a candle até o fim do arquivo.
O arquivo JSONL guarda um objeto por ciclo com snapshot completo da análise, origem dos dados e eventos de transição.

Se o bridge estiver em outra porta ou host, ajuste `MT5_HOST` e `MT5_PORT` no `.env`.

## Estrutura

- `app/config.py`: leitura das variáveis de ambiente
- `app/mt5_client.py`: conexão e leitura de candles no MT5 com suporte a `mt5linux`
- `app/indicators.py`: cálculo dos indicadores técnicos
- `app/analyzer.py`: consolidação da análise
- `app/main.py`: CLI e impressão do resumo final
