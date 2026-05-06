# MT5 LLM Agent

Agente de trading **stateful** que usa LLMs para tomar decisões em tempo real no MetaTrader 5.
A cada `ANALYSIS_INTERVAL` segundos, o pipeline coleta dados multi-timeframe, computa indicadores técnicos, testes estatísticos e modelos de volatilidade (GARCH/HAR-RV), injeta o estado da posição aberta no prompt e consulta a OpenAI - que responde com uma ação: `OPEN_LONG`, `OPEN_SHORT`, `HOLD`, `CLOSE` (sair antes de SL/TP) ou `TIGHTEN_STOP` (apertar SL para travar lucro).

```
MetaTrader 5 (MQL5 EA)
   │  GET /signal (poll periódico)
   ▼
FastAPI (Python) ──► OpenAI (json_schema strict)
   │                       │
   │  MT5 API + yfinance   └─► signals.jsonl
   ▼
 OHLCV multi-TF + posição aberta → contexto → prompt
```

---

## Instalação rápida

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

> `--host 0.0.0.0` é necessário se o MT5 roda no Wine, o EA não consegue acessar `127.0.0.1` sob Wine. Em Windows nativo, `127.0.0.1` funciona normalmente.

Depois de subir, abra `http://localhost:8000/` no navegador para o **dashboard de runtime** (equity curve, distribuição de ações, win rate por regime, latência do LLM, última decisão ao vivo - atualiza a cada 10 s).

Por padrão, os dados vêm do **yfinance** (`DATA_SOURCE=yfinance`).
Para usar dados do próprio broker via MT5, veja a seção abaixo.

---

## Configuração MT5 no Linux (fonte de dados MT5)

> Esta seção é opcional. Se preferir usar yfinance, pule aqui.

O pacote `mt5linux` cria uma ponte entre o Python Linux e o MT5 rodando no Wine/Bottles.
É necessário instalar Python **dentro** do ambiente Wine do MT5 e executar um servidor bridge.

### Passo 1 - Instalar Python dentro do Wine/Bottles

**Se usa Bottles:**
1. Abra o Bottles → entre no bottle onde o MT5 está instalado
2. Clique em **Run Executable** e selecione o instalador Python para Windows
   - Baixe: [python-3.11.9-amd64.exe](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
   - Instale em `drive_c\Python311` (marque "Add to PATH")

**Se usa Wine diretamente:**
```bash
wget https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
wine python-3.11.9-amd64.exe
```

### Passo 2 - Instalar MetaTrader5 e mt5linux dentro do Wine Python

No Bottles, abra o terminal do bottle: **Run** → **cmd** (ou **Terminal**).
No prompt `Z:\...>` que abrir, execute:

```
drive_c\users\steamuser\AppData\Local\Programs\Python\Python311\python.exe -m pip install MetaTrader5 mt5linux "numpy<2"
```

> Se instalou Python em outro caminho, ajuste o prefixo do exe.

> **Importante:** o `"numpy<2"` é necessário. NumPy 2.x usa funções C99 do
> Universal CRT (`crealf`, `cimagf`, etc) que o `ucrtbase.dll` do Wine ainda
> não implementa. Sem o downgrade, o bridge crasha assim que um cliente
> tenta conectar com erro `ucrtbase.dll.crealf, aborting`.
>
> Alternativa mais robusta: instalar `vcredist2019` ou `vcredist2022` via
> Bottles → aba **Dependencies** (substitui o ucrtbase do Wine pelo da
> Microsoft, que tem todas as funções C99).

### Passo 3 - Instalar mt5linux no Python Linux

No terminal Linux normal:
```bash
pip install mt5linux
```

### Passo 4 - Iniciar o bridge (terminal Wine, com MT5 aberto)

No mesmo terminal cmd do Bottles (**MT5 deve estar aberto e conectado ao broker**):

```
drive_c\users\steamuser\AppData\Local\Programs\Python\Python311\python.exe -m mt5linux --host 127.0.0.1 -p 18812
```

Você deve ver: `INFO SLAVE/18812[MainThread]: server started on [127.0.0.1]:18812`

O está bridge rodando.

### Passo 5 - Ativar no .env

```env
DATA_SOURCE=mt5
MT5_HOST=localhost
MT5_PORT=18812
```

O agente tenta MT5 primeiro. Se o bridge não estiver rodando, **cai automaticamente para yfinance** sem quebrar.

---

## Deploy do EA no MetaTrader 5

```bash
bash deploy_mql5.sh
```

O script copia `mql5/LLM_Trader.mq5` para o diretório `Experts` do MT5. _(Verifique o PATH do diretório `Experts`)_

Depois:
1. Abra o **MetaEditor** e compile o arquivo
2. Em **Tools > Options > Expert Advisors**, adicione a URL do servidor em *Allow WebRequest*:
   - Windows nativo: `http://127.0.0.1:8000`
   - Linux/Wine: `http://<IP-LAN>:8000` (ex: `http://192.168.0.10:8000`) - loopback falha sob Wine
3. Ajuste `InpServerURL` nos parâmetros do EA para a mesma URL liberada
4. Arraste o EA `LLM_Trader` para o gráfico do símbolo configurado

O **timeframe do gráfico não importa** - `OnTick()` dispara em todo tick, independente do TF exibido.

---

## Variáveis de ambiente (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `MODEL_VERSION` | `v1.6.0` | Tag de versão nos logs |
| `OPENAI_API_KEY` | - | Chave da OpenAI |
| `OPENAI_MODEL` | `gpt-4o` | Modelo OpenAI (suporta `json_schema` strict) |
| `SYMBOL` | `EURUSD` | Símbolo no broker (MT5) |
| `YF_SYMBOL` | `EURUSD=X` | Símbolo no yfinance |
| `DATA_SOURCE` | `yfinance` | `mt5` \| `yfinance` |
| `MT5_HOST` | `localhost` | Host do bridge mt5linux |
| `MT5_PORT` | `18812` | Porta do bridge mt5linux |
| `VOL_ESTIMATOR` | `garch` | `garch` \| `har_rv` \| `atr` |
| `PIP_SIZE` | `0.0001` | Tamanho do pip (EURUSD) |
| `ANALYSIS_INTERVAL` | `300` | Segundos entre análises |
| `MIN_CONFIDENCE` | `0.6` | Confiança mínima para agir |
| `RISK_PER_TRADE_PCT` | `1.0` | Risco-alvo por trade (% do equity) |
| `PIP_VALUE_PER_LOT` | `10.0` | USD por pip por 1 lote (EURUSD/USD ≈ 10) |
| `MIN_LOT` / `MAX_LOT` / `LOT_STEP` | `0.01` / `1.0` / `0.01` | Bounds e passo do broker |
| `DAILY_DRAWDOWN_PCT` | `2.0` | Circuit breaker - % do equity perdível por dia |
| `MAX_CONSECUTIVE_LOSSES` | `3` | Circuit breaker - perdas seguidas no dia |
| `MAX_TRADE_AGE_HOURS` | `24` | Time exit - força CLOSE em trade muito velho (0 desliga) |

---

## Zerar logs

Pra começar uma nova janela de coleta (ex.: depois de resetar o equity da conta demo):

```bash
python reset_logs.py            # interativo - exige digitar "SIM"
python reset_logs.py --yes      # pula confirmação
```

Apaga `signals.jsonl`, `trades.jsonl` e `trades_state.json`. Não toca em código, `.env`, paper ou posições do MT5. Reinicie o `uvicorn` em seguida pra limpar o `current_signal` em memória.

---

## Estrutura

```
mt5_llm_agent/
├── main.py                    # entry point FastAPI
├── config.py                  # settings via .env
├── agent/
│   ├── llm_manager.py         # OpenAI client (json_schema strict)
│   └── prompt_builder.py      # prompt hierárquico top-down + estado da posição
├── api/
│   ├── server.py              # endpoints /signal /health /analyze
│   └── dashboard.py           # GET / (HTML) + /api/dashboard (JSON agregado)
├── data/
│   ├── fetcher.py             # interface unificada MT5 + yfinance
│   ├── mt5_source.py          # OHLCV via MT5 (mt5linux ou MetaTrader5)
│   ├── yf_source.py           # fonte yfinance (fallback)
│   ├── positions.py           # leitura de posição aberta para o prompt stateful
│   └── trades.py              # captura do lifecycle real de trades fechados
├── features/
│   ├── context_builder.py     # monta contexto multi-TF
│   ├── indicators.py          # RSI, MACD, ATR, Bollinger, EMA
│   ├── statistics.py          # ADF, Ljung-Box, Hurst, VRT
│   ├── garch.py               # GARCH(1,1) - σ_{t+1} em pips
│   └── har_rv.py              # HAR-RV (Corsi) - σ_{t+1} em pips
├── risk/
│   ├── manager.py             # Signal + guardrails state-aware + pipeline de risco
│   ├── volatility.py          # estimador plugável GARCH > HAR-RV > ATR
│   ├── sizing.py              # position sizing dinâmico por confiança × vol inversa
│   └── circuit_breaker.py     # bloqueia novas aberturas em DD diário / perdas seguidas
├── analytics/
│   ├── simulator.py           # simulação post-hoc de TP/SL hit
│   └── metrics.py             # métricas por MODEL_VERSION
├── mql5/
│   └── LLM_Trader.mq5         # Expert Advisor MT5
├── paper/
│   └── paper.tex              # artigo acadêmico (WIP)
└── logs/
    ├── signals.jsonl          # log de sinais (decisões do LLM)
    └── trades.jsonl           # log de trades fechados (lifecycle real do MT5)
```

---

> Use conta **demo** apenas. Projeto experimental - sem garantias de performance.
