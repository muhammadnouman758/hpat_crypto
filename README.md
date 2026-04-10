# HPAT v6 — High-Precision Analytical Terminal

Enterprise-grade refactor of the HPAT trading terminal.

## Quick Start

```bash
pip install aiohttp websockets numpy python-dotenv

# Set credentials (never type them into the UI)
echo "BINANCE_API_KEY=your_key_here"     >> .env
echo "BINANCE_API_SECRET=your_secret_here" >> .env

python main.py
```

---

## Architecture

```
hpat_v6/
├── models.py            ← Typed state, EventBus, Decimal financial models
├── analytics_engine.py  ← All math — RSI/ATR (NumPy), VWAP, CVD, alerts
├── data_feed.py         ← asyncio WebSocket + aiohttp REST + exponential backoff
├── views/
│   ├── base_view.py     ← Base class: event subscription, canvas pool helpers
│   ├── orderbook_view.py← Object-pooled OB heatmap (no delete/redraw)
│   ├── vpvr_view.py     ← Object-pooled VPVR (no delete/redraw)
│   └── account_view.py  ← Account dashboard (event-driven, Decimal display)
├── main.py              ← MVVM coordinator, UI loop, hotkeys
└── requirements.txt
```

---

## What Changed vs v5

### 1 — Monolith → MVC/MVVM
- `HPAT_App` "God Object" (2 600 lines) decomposed into five focused modules.
- `models.py` uses `dataclasses` and `decimal.Decimal` for strict typing.
- `analytics_engine.py` owns all indicator math — zero UI imports.
- Each view class is independently instantiated; the main app only coordinates.

### 2 — Event-Driven Pub/Sub (no 250 ms polling for everything)
- `EventBus` in `models.py` replaces the 250 ms `after()` polling for UI panels.
- `OrderBookView`, `VPVRView`, `AccountView` subscribe to `'trade'`, `'acc_refresh'`,
  `'orderbook'` events emitted by `analytics_engine` and `data_feed`.
- The main 250 ms loop still handles the price display and analytics summary tab
  since those aggregate multiple state fields efficiently.

### 3 — asyncio + aiohttp + websockets
- Replaced blocking `urllib` and `threading.sleep` with full async I/O.
- `FeedController` owns a dedicated asyncio event loop in a background thread.
- WebSocket reconnection uses **exponential backoff with full jitter**:
  `delay = random.uniform(0, min(120, 1 * 2^attempt))` — prevents IP bans.
- Account REST calls are scheduled onto the feed loop thread-safely via
  `asyncio.run_coroutine_threadsafe()`.

### 4 — Object-Pooled Canvas Rendering
- `OrderBookView` and `VPVRView` pre-create all canvas items at `__init__`.
- On each tick only `canvas.coords()` and `canvas.itemconfig()` are called —
  **`canvas.delete('all')` is never called in the render path**.
- CPU usage for OB/VPVR redraws drops from O(n × create) to O(n × update).
- `base_view.py` provides `pool_rects()` / `pool_texts()` helpers for future views.

### 5 — Decimal Financial Arithmetic
- All P&L, position sizing, cost-basis, and commission calculations use
  `decimal.Decimal` with 28-digit precision (`getcontext().prec = 28`).
- `calc_pnl()` FIFO engine exclusively uses `Decimal` arithmetic.
- `Position.pnl()` is a typed method returning `Decimal`.
- Display conversion to `float` only happens at the final `fmt()` call.

### 6 — Secure Credential Management
- API keys are **never** entered in UI widgets or stored in global state at runtime.
- Credentials are loaded via `os.environ` or a `.env` file using `python-dotenv`.
- The account tab shows a read-only status; a CONNECT button triggers env loading.

### 7 — NumPy-Vectorised Indicators
- `calc_rsi()` uses `np.diff()` and array slicing — no Python `for` loop.
- `calc_atr()` still uses a short loop (14 candles) but is isolated for easy
  future NumPy migration or caching.

---

## Environment Variables

| Variable               | Description                     |
|------------------------|---------------------------------|
| `BINANCE_API_KEY`      | Binance spot/futures API key     |
| `BINANCE_API_SECRET`   | Binance API secret               |

Store in a `.env` file in the working directory (never commit to source control).

---

## Hotkeys

| Key | Action           |
|-----|------------------|
| 1–5 | Switch pair      |
| F   | Freeze/unfreeze tape |
| B   | Paper LONG       |
| S   | Paper SHORT      |
| X   | Close position   |

---

## Future: AI / ML Integration

The refactored state structure in `models.py` provides a clean "Feature Store" interface.
To wire an anomaly model:

```python
# In analytics_engine.py — replace hardcoded absorption threshold:
from sklearn.ensemble import IsolationForest

_iso_model = IsolationForest(contamination=0.05, random_state=42)

def fit_absorption_model() -> None:
    if len(STATE.cvd_history) < 100:
        return
    X = np.array([[p.cvd, p.price] for p in STATE.cvd_history])
    _iso_model.fit(X)

def check_absorption_ml(price: float) -> None:
    if len(STATE.cvd_history) < 30:
        return
    last = list(STATE.cvd_history)[-1]
    score = _iso_model.decision_function([[last.cvd, price]])[0]
    if score < -0.3:
        add_alert('ABSORB', f'⚠ ML ANOMALY score={score:.3f}', 'cyan')
```
