"""
HPAT v6 — models.py
Strict typed data models using dataclasses + Decimal for all financial values.
No raw floats for P&L, sizing, or cost-basis calculations.
"""

from __future__ import annotations
import collections
import threading
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Deque, Dict, List, Optional, Tuple
import datetime
import time

# Set Decimal precision for financial math
getcontext().prec = 28

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
PAIRS: Tuple[str, ...] = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT')

BASE_PRICES: Dict[str, Decimal] = {
    'BTCUSDT': Decimal('65000'),
    'ETHUSDT': Decimal('3200'),
    'SOLUSDT': Decimal('145'),
    'BNBUSDT': Decimal('580'),
    'XRPUSDT': Decimal('0.52'),
}

PAIR_LABELS: Dict[str, str] = {
    'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH',
    'SOLUSDT': 'SOL', 'BNBUSDT': 'BNB', 'XRPUSDT': 'XRP',
}

DECIMALS: Dict[str, int] = {
    'BTCUSDT': 2, 'ETHUSDT': 2, 'SOLUSDT': 3,
    'BNBUSDT': 2, 'XRPUSDT': 4,
}

# ─── THEME ────────────────────────────────────────────────────────────────────
C: Dict[str, str] = {
    'bg':      '#07090d',
    'bg1':     '#0e1420',
    'bg2':     '#161e2e',
    'bg3':     '#1e2a3c',
    'bg4':     '#263448',
    'border':  '#2e4060',
    'border2': '#3a5278',
    'border3': '#4a6490',
    'text':    '#ddeeff',
    'text2':   '#90b8d8',
    'text3':   '#607898',
    'green':   '#00f07a',
    'red':     '#ff3355',
    'amber':   '#ffc400',
    'blue':    '#29b8ff',
    'cyan':    '#00eeff',
    'teal':    '#1fffc0',
    'purple':  '#e040fb',
    'white':   '#ffffff',
}

# ─── TYPED RECORDS ────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    price:  Decimal
    qty:    Decimal
    vol:    Decimal
    is_buy: bool
    tier:   str
    ts:     str  # formatted timestamp string

@dataclass
class OHLCVCandle:
    t:      int   # bucket timestamp ms
    o:      float
    h:      float
    l:      float
    c:      float
    v:      float
    closed: bool = False

@dataclass
class CVDPoint:
    cvd:   float
    price: float
    t:     int

@dataclass
class OIPoint:
    oi: float
    t:  int

@dataclass
class PricePoint:
    price: float
    t:     int

@dataclass
class AlertRecord:
    type:  str
    msg:   str
    color: str
    ts:    str

@dataclass
class LiqEvent:
    side:  str
    size:  Decimal
    price: Decimal
    ts:    str

@dataclass
class Position:
    """Paper trading position — all financial values in Decimal."""
    side:  str           # 'LONG' | 'SHORT'
    entry: Decimal
    size:  Decimal       # notional in USDT
    t:     int           # open timestamp ms

    def pnl(self, current_price: Decimal) -> Decimal:
        if self.side == 'LONG':
            return (current_price - self.entry) / self.entry * self.size
        else:
            return (self.entry - current_price) / self.entry * self.size

# ─── MARKET STATE ─────────────────────────────────────────────────────────────

class MarketState:
    """
    Thread-safe market state.
    Background threads write via lock-guarded setters.
    UI reads snapshots — never holds the lock during rendering.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # ── Identity ──────────────────────────────────────────────────────────
        self.pair: str = 'BTCUSDT'

        # ── Prices (float OK for display; Decimal used for calculations) ──────
        self.price:      float = 0.0
        self.bid:        float = 0.0
        self.ask:        float = 0.0
        self.prev_price: float = 0.0

        # ── Volume flow ───────────────────────────────────────────────────────
        self.cvd:   float = 0.0
        self.vbuy:  float = 0.0
        self.vsell: float = 0.0

        # ── VWAP accumulators ─────────────────────────────────────────────────
        self.vwap_sum_pv:  float = 0.0
        self.vwap_sum_v:   float = 0.0
        self.vwap_sum_pv2: float = 0.0
        self.vwap_val:     float = 0.0
        self.vwap_bands:   Dict[str, float] = {'up1': 0, 'up2': 0, 'dn1': 0, 'dn2': 0}

        # ── RSI ───────────────────────────────────────────────────────────────
        self.rsi_1m: float = 50.0
        self.rsi_5m: float = 50.0

        # ── OI / Funding ──────────────────────────────────────────────────────
        self.oi:         float = 0.0
        self.funding:    float = 0.0
        self.funding_cd: int   = 0
        self.oi_history:      Deque[OIPoint]   = collections.deque(maxlen=200)
        self.funding_history: Deque[float]     = collections.deque(maxlen=20)

        # ── Order book ────────────────────────────────────────────────────────
        self.ob_bids: List = []
        self.ob_asks: List = []

        # ── Volume profile ────────────────────────────────────────────────────
        self.vpvr:     Dict[float, float] = {}
        self.footprint: Dict[float, Dict[str, float]] = {}
        self.poc:       float = 0.0

        # ── Trade tape ────────────────────────────────────────────────────────
        self.trades:      Deque[TradeRecord] = collections.deque(maxlen=50)
        self.trade_win:   Deque[Dict]        = collections.deque(maxlen=200)
        self.tape_frozen: bool = False
        self.tape_buffer: Deque[TradeRecord] = collections.deque(maxlen=200)

        # ── Price history buffers ─────────────────────────────────────────────
        self.atr_prices:    Deque[PricePoint] = collections.deque(maxlen=2000)
        self.atr5m_prices:  Deque[PricePoint] = collections.deque(maxlen=500)
        self.cvd_history:   Deque[CVDPoint]   = collections.deque(maxlen=200)
        self.ring_buf:      Deque[float]       = collections.deque(maxlen=1000)

        # ── Session H/L ───────────────────────────────────────────────────────
        self.high30:  float = 0.0
        self.low30:   float = float('inf')
        self.adr_high: float = 0.0
        self.adr_low:  float = float('inf')

        # ── Alerts ────────────────────────────────────────────────────────────
        self.alerts:          Deque[AlertRecord] = collections.deque(maxlen=20)
        self.alert_count:     int  = 0
        self.alert_cooldown:  Dict[str, int] = {}
        self.absorption_alerted: bool = False

        # ── Paper position ────────────────────────────────────────────────────
        self.position: Optional[Position] = None

        # ── Liquidations ──────────────────────────────────────────────────────
        self.liq_total:  Decimal = Decimal('0')
        self.liq_events: Deque[LiqEvent] = collections.deque(maxlen=10)

        # ── Multi-pair dominance ──────────────────────────────────────────────
        self.dm_prices:  Dict[str, float] = {p: 0.0 for p in PAIRS}
        self.dm_cvd:     Dict[str, float] = {p: 0.0 for p in PAIRS}
        self.dm_obi:     Dict[str, float] = {p: 0.0 for p in PAIRS}
        self.corr_prices: Dict[str, Deque[PricePoint]] = {
            p: collections.deque(maxlen=200) for p in PAIRS
        }

        # ── Candles ───────────────────────────────────────────────────────────
        self.candles: Dict[str, Dict[str, Deque[OHLCVCandle]]] = {
            p: {'1m': collections.deque(maxlen=100),
                '5m': collections.deque(maxlen=100),
                '30m': collections.deque(maxlen=100)}
            for p in PAIRS
        }

        # ── Sim ───────────────────────────────────────────────────────────────
        self.sim_running: bool = False
        self.ws_status:   str  = 'INIT'
        self.base_prices: Dict[str, Decimal] = dict(BASE_PRICES)

    def snapshot(self) -> 'MarketState':
        """Return a shallow copy for safe UI reads (avoids holding the lock)."""
        with self._lock:
            # Create a new object and copy scalar + mutable-safe attributes.
            # Deques and dicts are snapshotted by converting to list/dict copies.
            s = object.__new__(MarketState)
            s.__dict__.update(self.__dict__)
            # Shallow-freeze the collections that the analytics engine may mutate
            s.ob_bids = list(self.ob_bids)
            s.ob_asks = list(self.ob_asks)
            s.trades  = collections.deque(self.trades, maxlen=self.trades.maxlen)
            s.vpvr    = dict(self.vpvr)
            s.footprint = {k: dict(v) for k, v in self.footprint.items()}
            s.cvd_history  = collections.deque(self.cvd_history, maxlen=self.cvd_history.maxlen)
            s.atr_prices   = collections.deque(self.atr_prices,  maxlen=self.atr_prices.maxlen)
            s.atr5m_prices = collections.deque(self.atr5m_prices,maxlen=self.atr5m_prices.maxlen)
            s.oi_history   = collections.deque(self.oi_history,  maxlen=self.oi_history.maxlen)
            s.trade_win    = collections.deque(self.trade_win,   maxlen=self.trade_win.maxlen)
            s.dm_prices    = dict(self.dm_prices)
            s.dm_cvd       = dict(self.dm_cvd)
            s.dm_obi       = dict(self.dm_obi)
            s.corr_prices  = {p: collections.deque(dq) for p, dq in self.corr_prices.items()}
            s.alerts       = collections.deque(self.alerts, maxlen=self.alerts.maxlen)
            s.liq_events   = collections.deque(self.liq_events, maxlen=self.liq_events.maxlen)
            s.candles      = {
                p: {tf: collections.deque(dq) for tf, dq in tfs.items()}
                for p, tfs in self.candles.items()
            }
            return s

    def reset_pair(self, pair: str) -> None:
        with self._lock:
            self.pair       = pair
            self.price      = 0.0; self.prev_price = 0.0
            self.bid        = 0.0; self.ask = 0.0
            self.cvd        = 0.0; self.vbuy = 0.0; self.vsell = 0.0
            self.vpvr       = {}; self.footprint = {}
            self.high30     = 0.0; self.low30 = float('inf')
            self.adr_high   = 0.0; self.adr_low = float('inf')
            self.oi         = 0.0; self.oi_history.clear()
            self.funding    = 0.0; self.funding_cd = 0; self.funding_history.clear()
            self.cvd_history.clear(); self.atr_prices.clear(); self.atr5m_prices.clear()
            self.trades.clear(); self.trade_win.clear(); self.tape_buffer.clear()
            self.ob_bids    = []; self.ob_asks = []
            self.ring_buf.clear()
            self.vwap_sum_pv = 0; self.vwap_sum_v = 0; self.vwap_sum_pv2 = 0; self.vwap_val = 0
            self.rsi_1m     = 50.0; self.rsi_5m = 50.0
            self.poc        = 0.0
            self.liq_total  = Decimal('0'); self.liq_events.clear()
            self.alerts.clear(); self.alert_cooldown = {}
            self.absorption_alerted = False
            self.dm_prices[pair] = float(self.base_prices.get(pair, BASE_PRICES[pair]))
            self.dm_cvd[pair]    = 0.0
            self.dm_obi[pair]    = 0.0
            self.corr_prices[pair].clear()


# ─── ACCOUNT STATE ────────────────────────────────────────────────────────────

@dataclass
class Balance:
    asset:   str
    free:    Decimal
    locked:  Decimal
    total:   Decimal
    usd_val: Decimal

@dataclass
class OpenOrder:
    symbol:   str
    side:     str
    type:     str
    price:    Decimal
    qty:      Decimal
    filled:   Decimal
    status:   str
    time:     str
    order_id: int

@dataclass
class TradeHistoryEntry:
    symbol:     str
    side:       str
    price:      Decimal
    qty:        Decimal
    quoteQty:   Decimal
    commission: Decimal
    commAsset:  str
    time:       str
    time_ms:    int
    pnl:        Optional[Decimal] = None

class AccountState:
    def __init__(self) -> None:
        # Credentials — never stored in plain global state in production;
        # loaded from env/.env at startup only.
        self.api_key:    str = ''
        self.api_secret: str = ''

        self.connected: bool = False
        self.error:     str  = ''

        self.balances:      List[Balance]           = []
        self.total_usd:     Decimal                 = Decimal('0')
        self.open_orders:   List[OpenOrder]         = []
        self.trade_history: List[TradeHistoryEntry] = []
        self.realized_pnl:  Dict[str, Decimal]      = {}
        self.total_pnl:     Decimal                 = Decimal('0')
        self.win_trades:    int = 0
        self.loss_trades:   int = 0
        self.best_trade:    Optional[TradeHistoryEntry] = None
        self.worst_trade:   Optional[TradeHistoryEntry] = None
        self.last_refresh:  float = 0.0
        self.refreshing:    bool  = False


# ─── EVENT BUS ────────────────────────────────────────────────────────────────

class Event:
    __slots__ = ('kind', 'payload')
    def __init__(self, kind: str, payload=None):
        self.kind    = kind
        self.payload = payload

class EventBus:
    """
    Simple synchronous Pub/Sub event bus.
    Publishers call emit(); subscribers register with subscribe().
    All callbacks execute on the emitting thread — lightweight and zero-dependency.
    UI callbacks must be scheduled via Tk.after() if they touch widgets.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List] = collections.defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, kind: str, callback) -> None:
        with self._lock:
            self._subscribers[kind].append(callback)

    def emit(self, event: Event) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event.kind, []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def subscribe_many(self, mapping: Dict[str, object]) -> None:
        for kind, cb in mapping.items():
            self.subscribe(kind, cb)


# ─── SINGLETON INSTANCES ──────────────────────────────────────────────────────
# These are imported by all other modules.

STATE = MarketState()
ACC   = AccountState()
BUS   = EventBus()
