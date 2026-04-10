"""
HPAT v6 — gemini_engine.py
Real-time AI prediction engine powered by Google Gemini.

Design principles:
  • Every 10 seconds, a rich feature vector is extracted from live market state
    and encoded into a structured, expert-level prompt using multi-role prompt
    engineering (System → Expert Persona → Market Context → Task → Output Schema).
  • Responses are parsed into a typed PredictionResult dataclass.
  • The engine runs entirely on the asyncio feed loop thread — zero Tkinter
    coupling. Results are emitted via EventBus so any UI panel can subscribe.
  • Rate-limit aware: skips a cycle if the previous call is still in-flight.
  • API key loaded from GEMINI_API_KEY environment variable.
  • Full conversation history is NOT kept (stateless per-call) to keep latency
    low; instead, rich context is embedded in every prompt.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import STATE, BUS, Event, DECIMALS, PAIR_LABELS, C
from analytics_engine import (
    calc_atr, calc_rsi, calc_obi, get_vwap, get_poc,
    market_regime, composite_signal, fmt, fmt_k, now_ms,
)

# ─── RESULT MODEL ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    pair:            str
    timestamp:       str

    # Directional call
    direction:       str    # 'LONG' | 'SHORT' | 'NEUTRAL' | 'NO_EDGE'
    conviction:      int    # 1–10
    time_horizon:    str    # e.g. '30–90 seconds'

    # Price targets (Decimal-safe strings)
    entry_zone_lo:   float
    entry_zone_hi:   float
    stop_loss:       float
    take_profit_1:   float
    take_profit_2:   float
    risk_reward:     float

    # Reasoning
    primary_driver:  str    # one-liner: what's causing the signal
    confluence:      List[str]   # bullet list of supporting factors
    invalidation:    str    # what price action would invalidate the call
    regime:          str    # market regime label

    # Risk
    risk_level:      str    # 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME'
    position_size_pct: float  # % of capital suggested (Kelly-adjusted)

    # Raw model output for debugging
    raw_json:        Dict    = field(default_factory=dict)
    error:           Optional[str] = None


# ─── FEATURE EXTRACTOR ────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a rich, structured feature snapshot from the live MarketState.
    All values are converted to plain Python types (no Decimal/deque) so they
    can be JSON-serialised and embedded directly in the prompt.
    """

    @staticmethod
    def extract(pair: str) -> Dict:
        s   = STATE.snapshot()
        dec = DECIMALS.get(pair, 2)
        now = now_ms()

        # ── Price & spread ─────────────────────────────────────────────────
        price  = s.price
        bid    = s.bid
        ask    = s.ask
        spread = ask - bid if ask and bid else 0.0
        spread_bps = (spread / max(bid, 1)) * 10_000

        # ── VWAP distance ──────────────────────────────────────────────────
        vwap, up1, up2, dn1, dn2 = get_vwap()
        vwap_dist_pct = (price - vwap) / max(vwap, 1) * 100 if vwap else 0.0
        vwap_band = ("above_2sigma" if price > up2 else
                     "above_1sigma" if price > up1 else
                     "below_1sigma" if price < dn1 else
                     "below_2sigma" if price < dn2 else "inside_band")

        # ── RSI ────────────────────────────────────────────────────────────
        prices_1m = [x.price for x in list(s.atr5m_prices)[-30:]]
        rsi_1m = calc_rsi(prices_1m) if len(prices_1m) > 15 else 50.0
        rsi_5m = calc_rsi(prices_1m[::2]) if len(prices_1m) > 20 else rsi_1m

        # ── ATR / Volatility ───────────────────────────────────────────────
        atr_5m  = calc_atr(pair, '5m')
        atr_30m = calc_atr(pair, '30m')
        vol_ratio = atr_5m / max(atr_30m, 1e-9) if atr_5m and atr_30m else 1.0
        atr_pct   = atr_5m / max(price, 1) * 100 if atr_5m and price else 0.0

        # ── CVD & order flow ───────────────────────────────────────────────
        cvd      = s.cvd
        vbuy     = s.vbuy
        vsell    = s.vsell
        total_vol = vbuy + vsell or 1.0
        buy_dom   = vbuy / total_vol * 100

        # CVD slope: last 30 points
        cvd_hist = list(s.cvd_history)
        cvd_slope = 0.0
        if len(cvd_hist) >= 10:
            recent   = [p.cvd for p in cvd_hist[-10:]]
            cvd_slope = (recent[-1] - recent[0]) / max(abs(recent[0]), 1.0)

        # CVD divergence flag
        price_hist   = [p.price for p in list(s.atr_prices)[-10:]]
        cvd_diverge  = 'none'
        if len(price_hist) >= 10 and len(cvd_hist) >= 10:
            p_up  = price_hist[-1] > price_hist[0]
            c_up  = cvd_hist[-1].cvd > cvd_hist[0].cvd
            if p_up and not c_up:  cvd_diverge = 'bearish'
            elif not p_up and c_up: cvd_diverge = 'bullish'

        # ── Order book ─────────────────────────────────────────────────────
        obi = calc_obi(s.ob_bids, s.ob_asks)

        # Top 5 bid/ask levels
        top_bids = [[round(float(b[0]), dec), round(float(b[1]), 4)]
                    for b in s.ob_bids[:5]] if s.ob_bids else []
        top_asks = [[round(float(a[0]), dec), round(float(a[1]), 4)]
                    for a in s.ob_asks[:5]] if s.ob_asks else []

        # Detect bid/ask walls (>3× avg size)
        if top_bids:
            avg_bid_sz = sum(b[1] for b in top_bids) / len(top_bids)
            bid_walls  = [b[0] for b in top_bids if b[1] > avg_bid_sz * 3]
        else:
            bid_walls = []
        if top_asks:
            avg_ask_sz = sum(a[1] for a in top_asks) / len(top_asks)
            ask_walls  = [a[0] for a in top_asks if a[1] > avg_ask_sz * 3]
        else:
            ask_walls = []

        # ── POC / VPVR ─────────────────────────────────────────────────────
        poc, poc_dist = get_poc()
        poc_side = 'above' if price > poc else 'below' if poc else 'unknown'

        # Value area: top 70% of VPVR by volume
        vah, val_price = 0.0, 0.0
        if s.vpvr:
            total_v   = sum(s.vpvr.values())
            target_v  = total_v * 0.70
            sorted_bins = sorted(s.vpvr.items(), key=lambda x: x[1], reverse=True)
            acc_v = 0.0
            value_bins = []
            for b, v in sorted_bins:
                acc_v += v
                value_bins.append(b)
                if acc_v >= target_v:
                    break
            if value_bins:
                vah      = max(value_bins)
                val_price = min(value_bins)

        # ── Recent candles summary ─────────────────────────────────────────
        candles_5m  = list(s.candles[pair]['5m'])[-5:]
        candles_1m  = list(s.candles[pair]['1m'])[-5:]
        candle_summary = []
        for c in candles_5m:
            body     = abs(c.c - c.o)
            wick_top = c.h - max(c.o, c.c)
            wick_bot = min(c.o, c.c) - c.l
            direction_c = 'bullish' if c.c > c.o else 'bearish'
            candle_summary.append({
                'tf': '5m', 'dir': direction_c,
                'body_pct': round(body / max(c.o, 1) * 100, 4),
                'wick_top_pct': round(wick_top / max(c.o, 1) * 100, 4),
                'wick_bot_pct': round(wick_bot / max(c.o, 1) * 100, 4),
                'vol': round(c.v, 2),
            })

        # ── OI / Funding ───────────────────────────────────────────────────
        oi      = s.oi
        funding = s.funding
        oi_hist = list(s.oi_history)
        oi_chg_1h = 0.0
        if len(oi_hist) >= 2:
            h1 = [x for x in oi_hist if now - x.t < 3_600_000]
            if len(h1) >= 2:
                oi_chg_1h = (oi - h1[0].oi) / max(h1[0].oi, 1) * 100

        # ── Recent large trades ────────────────────────────────────────────
        large_trades = []
        for t in list(s.trades)[:10]:
            if t.tier in ('WHALE', 'INST'):
                large_trades.append({
                    'side': 'BUY' if t.is_buy else 'SELL',
                    'tier': t.tier,
                    'vol_usd': round(float(t.vol), 0),
                    'price': round(float(t.price), dec),
                    'ts': t.ts,
                })

        # ── Session stats ──────────────────────────────────────────────────
        session_range = s.adr_high - s.adr_low if s.adr_high and s.adr_low != float('inf') else 0
        range_completion = abs(price - s.adr_low) / max(session_range, 1) * 100 if session_range else 50.0

        # ── Market regime ──────────────────────────────────────────────────
        regime_label, _ = market_regime(pair)
        comp_signal, _  = composite_signal()

        # ── Multi-pair context ─────────────────────────────────────────────
        btc_price = s.dm_prices.get('BTCUSDT', 0.0)
        btc_cvd   = s.dm_cvd.get('BTCUSDT', 0.0)
        btc_obi   = s.dm_obi.get('BTCUSDT', 0.0)

        return {
            "pair": pair,
            "label": PAIR_LABELS.get(pair, pair),
            "timestamp_utc": datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),

            "price": {
                "current": round(price, dec),
                "bid": round(bid, dec),
                "ask": round(ask, dec),
                "spread_bps": round(spread_bps, 4),
                "session_high": round(s.adr_high, dec) if s.adr_high else None,
                "session_low": round(s.adr_low, dec) if s.adr_low != float('inf') else None,
                "session_range_pct": round(session_range / max(price, 1) * 100, 4),
                "range_completion_pct": round(range_completion, 2),
            },

            "momentum": {
                "rsi_1m": round(rsi_1m, 2),
                "rsi_5m": round(rsi_5m, 2),
                "cvd": round(cvd, 2),
                "cvd_slope_10tick": round(cvd_slope, 6),
                "cvd_divergence": cvd_diverge,
                "buy_volume_pct": round(buy_dom, 2),
                "composite_signal": comp_signal,
            },

            "volatility": {
                "atr_5m": round(atr_5m, dec + 2) if atr_5m else None,
                "atr_30m": round(atr_30m, dec + 2) if atr_30m else None,
                "atr_pct": round(atr_pct, 4),
                "vol_expansion_ratio": round(vol_ratio, 4),
                "regime": regime_label,
            },

            "vwap": {
                "value": round(vwap, dec) if vwap else None,
                "distance_pct": round(vwap_dist_pct, 4),
                "band_position": vwap_band,
                "upper_1sigma": round(up1, dec) if up1 else None,
                "upper_2sigma": round(up2, dec) if up2 else None,
                "lower_1sigma": round(dn1, dec) if dn1 else None,
                "lower_2sigma": round(dn2, dec) if dn2 else None,
            },

            "volume_profile": {
                "poc": round(poc, dec) if poc else None,
                "poc_distance_pct": round(poc_dist, 4),
                "poc_side": poc_side,
                "value_area_high": round(vah, dec) if vah else None,
                "value_area_low": round(val_price, dec) if val_price else None,
            },

            "order_book": {
                "obi": round(obi, 4),
                "top_bids": top_bids,
                "top_asks": top_asks,
                "bid_walls": bid_walls,
                "ask_walls": ask_walls,
            },

            "candles": candle_summary,

            "derivatives": {
                "open_interest_usd": round(oi, 0) if oi else None,
                "oi_change_1h_pct": round(oi_chg_1h, 4),
                "funding_rate_pct": round(funding, 6),
                "funding_bias": ("longs_paying" if funding > 0 else
                                  "shorts_paying" if funding < 0 else "neutral"),
            },

            "large_trades": large_trades,

            "market_context": {
                "btc_price": round(btc_price, 2) if btc_price else None,
                "btc_cvd_direction": "positive" if btc_cvd > 0 else "negative",
                "btc_obi": round(btc_obi, 4),
                "is_btc_pair": pair == 'BTCUSDT',
            },
        }


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds a multi-stage, expert-level prompt using structured prompt
    engineering best practices:

    1. SYSTEM PERSONA    — establishes the AI's role and epistemic constraints
    2. DOMAIN FRAMING    — defines what high-quality analysis looks like
    3. MARKET DATA       — structured JSON feature vector
    4. ANALYTICAL TASK   — precise, step-by-step reasoning instructions
    5. OUTPUT CONTRACT   — strict JSON schema with field descriptions
    6. CALIBRATION RULES — explicit biases to avoid, edge-case handling
    """

    SYSTEM_PERSONA = """\
You are HPAT-AI, a senior quantitative analyst and market microstructure expert \
with 15 years of experience in crypto derivatives trading, order flow analysis, \
and real-time signal generation. You specialise in ultra-short-term (10–120 second) \
directional calls derived from order flow, volume profile, and momentum convergence.

Your analysis is grounded in:
- Market microstructure: order book depth, bid-ask imbalance, spoofing detection
- Order flow: CVD divergence, aggressive vs. passive flow, absorption patterns
- Volume profile: POC magnetism, value area breakouts/rejections
- Momentum: multi-timeframe RSI, VWAP mean-reversion, ATR-relative volatility
- Derivatives sentiment: funding rate extremes, OI build vs. liquidation cascades

You are not a chatbot. You produce only structured JSON output. Every call must be \
internally consistent (e.g., a LONG call must have TP > entry > SL).\
"""

    DOMAIN_FRAMING = """\
HIGH-QUALITY ANALYSIS CRITERIA:
• Conviction 8–10 requires at least 3 independent, non-correlated confirming factors
• Conviction 4–7 means 2 confirming factors with one ambiguous factor
• Conviction 1–3 or NO_EDGE means conflicting signals or insufficient data
• NEVER force a directional call when indicators conflict — use NO_EDGE
• A bearish CVD divergence while price is above VWAP is a HIGH-WEIGHT bearish signal
• OBI > +0.3 with rising CVD = strong immediate buy pressure
• Funding > +0.05% is a crowding warning for longs — reduces long conviction
• POC magnetism: price within 0.1% of POC usually signals consolidation, not trend
• Extreme RSI (>75 or <25) on 1m frame = mean-reversion bias, not continuation\
"""

    OUTPUT_SCHEMA = """\
Return ONLY valid JSON. No markdown fences, no preamble, no trailing text.

{
  "direction": "LONG" | "SHORT" | "NEUTRAL" | "NO_EDGE",
  "conviction": <integer 1-10>,
  "time_horizon": "<e.g. 30-60 seconds>",
  "entry_zone": {"lo": <float>, "hi": <float>},
  "stop_loss": <float>,
  "take_profit_1": <float>,
  "take_profit_2": <float>,
  "risk_reward": <float, rounded to 2dp>,
  "primary_driver": "<one sentence, max 120 chars>",
  "confluence": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "invalidation": "<price action that kills this call, max 100 chars>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "EXTREME",
  "position_size_pct": <float 0.0-5.0, Kelly-adjusted for conviction and risk>,
  "regime": "<TRENDING | RANGING | VOLATILE>",
  "reasoning_chain": [
    "<step 1: assess momentum>",
    "<step 2: assess order flow>",
    "<step 3: assess structure>",
    "<step 4: convergence verdict>"
  ]
}\
"""

    CALIBRATION_RULES = """\
CALIBRATION RULES (follow strictly):
1. If spread_bps > 5.0, increase risk_level by one grade (LOW→MEDIUM etc.)
2. If vol_expansion_ratio > 2.0, time_horizon must be ≤ 30 seconds
3. If funding_rate_pct > 0.08%, cap LONG conviction at 6 max
4. If funding_rate_pct < -0.08%, cap SHORT conviction at 6 max
5. If cvd_divergence == 'bearish' and direction == 'LONG', conviction ≤ 5
6. If cvd_divergence == 'bullish' and direction == 'SHORT', conviction ≤ 5
7. stop_loss must be at least 0.5× ATR away from entry_zone midpoint
8. take_profit_1 must give RR ≥ 1.5; take_profit_2 must give RR ≥ 2.5
9. position_size_pct = base_kelly × (conviction/10) × (1 / risk_multiplier)
   where risk_multiplier = 1 (LOW), 1.5 (MEDIUM), 2.5 (HIGH), 4 (EXTREME)
10. If fewer than 3 candles available in any timeframe, reduce conviction by 2\
"""

    @classmethod
    def build(cls, features: Dict) -> str:
        pair      = features['pair']
        label     = features['label']
        ts        = features['timestamp_utc']
        price     = features['price']['current']
        atr_5m    = features['volatility']['atr_5m'] or 0
        atr_pct   = features['volatility']['atr_pct']

        # Dynamic price context injected into the task section
        approx_stop_range = round(atr_5m * 1.5, DECIMALS.get(pair, 2))
        approx_tp1        = round(atr_5m * 2.0, DECIMALS.get(pair, 2))

        task = f"""\
ANALYTICAL TASK — {label}/USDT @ {price} UTC {ts}

Step through the reasoning chain:
1. MOMENTUM ASSESSMENT: Evaluate RSI_{'{1m}'}, RSI_{'{5m}'}, CVD slope, and buy dominance. \
   Are they aligned? Note any divergences.
2. ORDER FLOW ASSESSMENT: Evaluate OBI, bid/ask walls, and recent whale/inst trades. \
   Is aggressive flow directional or two-sided?
3. STRUCTURE ASSESSMENT: Where is price relative to VWAP bands, POC, and value area? \
   Is price in an area of high or low liquidity?
4. DERIVATIVES OVERLAY: Does the funding rate create a crowding risk? \
   Is OI expanding or contracting with this move?
5. CONVERGENCE VERDICT: Do momentum, flow, and structure align? \
   State direction, conviction (1–10), and the single highest-weight reason.

ATR context: 5m ATR ≈ {atr_5m} ({atr_pct:.3f}% of price).
Suggested stop distance ≈ {approx_stop_range} (1.5× ATR).
Suggested TP1 distance ≈ {approx_tp1} (2.0× ATR).

MARKET DATA (JSON):
{json.dumps(features, indent=2, default=str)}
"""

        full_prompt = "\n\n---\n\n".join([
            cls.SYSTEM_PERSONA,
            cls.DOMAIN_FRAMING,
            task,
            cls.OUTPUT_SCHEMA,
            cls.CALIBRATION_RULES,
        ])
        return full_prompt


# ─── RESPONSE PARSER ──────────────────────────────────────────────────────────

class ResponseParser:

    @staticmethod
    def parse(raw_text: str, features: Dict) -> PredictionResult:
        pair = features['pair']
        ts   = features['timestamp_utc']
        dec  = DECIMALS.get(pair, 2)
        price = features['price']['current']

        # Strip any accidental markdown fences
        cleaned = re.sub(r'```(?:json)?', '', raw_text).strip()
        # Extract first JSON object
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            return ResponseParser._error_result(pair, ts, "No JSON found in response", price)

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            return ResponseParser._error_result(pair, ts, f"JSON parse error: {e}", price)

        try:
            entry = data.get('entry_zone', {})
            lo    = float(entry.get('lo', price))
            hi    = float(entry.get('hi', price))

            result = PredictionResult(
                pair=pair,
                timestamp=ts,
                direction=str(data.get('direction', 'NO_EDGE')).upper(),
                conviction=max(1, min(10, int(data.get('conviction', 1)))),
                time_horizon=str(data.get('time_horizon', 'N/A')),
                entry_zone_lo=round(lo, dec),
                entry_zone_hi=round(hi, dec),
                stop_loss=round(float(data.get('stop_loss', price)), dec),
                take_profit_1=round(float(data.get('take_profit_1', price)), dec),
                take_profit_2=round(float(data.get('take_profit_2', price)), dec),
                risk_reward=round(float(data.get('risk_reward', 0.0)), 2),
                primary_driver=str(data.get('primary_driver', ''))[:200],
                confluence=list(data.get('confluence', []))[:5],
                invalidation=str(data.get('invalidation', ''))[:200],
                regime=str(data.get('regime', 'UNKNOWN')),
                risk_level=str(data.get('risk_level', 'HIGH')).upper(),
                position_size_pct=round(float(data.get('position_size_pct', 0.0)), 2),
                raw_json=data,
            )
            return result

        except (KeyError, TypeError, ValueError) as e:
            return ResponseParser._error_result(pair, ts, f"Field error: {e}", price)

    @staticmethod
    def _error_result(pair: str, ts: str, msg: str, price: float) -> PredictionResult:
        return PredictionResult(
            pair=pair, timestamp=ts,
            direction='NO_EDGE', conviction=0, time_horizon='N/A',
            entry_zone_lo=price, entry_zone_hi=price,
            stop_loss=price, take_profit_1=price, take_profit_2=price,
            risk_reward=0.0,
            primary_driver='Prediction error',
            confluence=[], invalidation='N/A',
            regime='UNKNOWN', risk_level='EXTREME',
            position_size_pct=0.0,
            error=msg,
        )


# ─── GEMINI CLIENT ────────────────────────────────────────────────────────────

class GeminiEngine:
    """
    Async Gemini prediction engine.
    - Polls every 10 seconds via asyncio.sleep (never blocking)
    - Uses google-generativeai SDK with gemini-2.0-flash (low latency)
    - Emits 'ai_prediction' events on the EventBus after each cycle
    - Thread-safe: all state is owned by the asyncio loop thread
    """

    MODEL_NAME   = 'gemini-2.5-flash'
    POLL_SECONDS = 300

    def __init__(self) -> None:
        self._api_key:    Optional[str] = None
        self._client      = None
        self._in_flight:  bool = False
        self._last_result: Optional[PredictionResult] = None
        self._call_count: int = 0
        self._error_count: int = 0
        self._last_latency_ms: float = 0.0
        self.enabled:     bool = False

    def configure(self, api_key: str) -> bool:
        """Load the Gemini API key and instantiate the client."""
        if not api_key:
            print("[DEBUG] FAILED: API key is empty or None.") # <-- ADD THIS
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client  = genai.GenerativeModel(
                model_name=self.MODEL_NAME,
                generation_config={
                    'temperature':      0.15,   # low temperature = deterministic analysis
                    'top_p':            0.85,
                    'top_k':            32,
                    'max_output_tokens': 1024,
                },
                safety_settings=[
                    {'category': 'HARM_CATEGORY_HARASSMENT',        'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_HATE_SPEECH',        'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',  'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',  'threshold': 'BLOCK_NONE'},
                ],
            )
            self._api_key = api_key
            self.enabled  = True
            print("[DEBUG] Gemini successfully configured and enabled.")
            return True
        except Exception as e:
            self.enabled = False
            print(f"[DEBUG] Gemini initialization ERROR: {e}")
            return False

    async def run(self, stop_event: asyncio.Event) -> None:
        """Main coroutine — runs on the feed asyncio loop."""
        if not self.enabled or not self._client:
            return
        # Stagger first call by 5 s to let the feed warm up
        await asyncio.sleep(5)
        while not stop_event.is_set():
            if not self._in_flight:
                asyncio.create_task(self._predict_cycle())
            await asyncio.sleep(self.POLL_SECONDS)

    async def _predict_cycle(self) -> None:
        self._in_flight = True
        pair = STATE.pair
        t0   = time.monotonic()
        try:
            features = FeatureExtractor.extract(pair)
            prompt   = PromptBuilder.build(features)
            result   = await self._call_gemini(prompt, features)
            self._last_latency_ms = (time.monotonic() - t0) * 1000
            self._last_result = result
            self._call_count += 1
            if result.error:
                self._error_count += 1
            BUS.emit(Event('ai_prediction', result))
        except Exception as e:
            print(f"[DEBUG] Gemini Prediction Cycle ERROR: {e}") # <-- ADD THIS
            self._error_count += 1
            BUS.emit(Event('ai_prediction_error', str(e)))
        finally:
            self._in_flight = False

    async def _call_gemini(self, prompt: str, features: Dict) -> PredictionResult:
        """Async wrapper around the synchronous Gemini SDK call."""
        loop = asyncio.get_event_loop()
        # Run the blocking SDK call in a thread pool so we never block the loop
        raw_text = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content(prompt).text
        )
        return ResponseParser.parse(raw_text, features)

    @property
    def stats(self) -> Dict:
        return {
            'calls':       self._call_count,
            'errors':      self._error_count,
            'latency_ms':  round(self._last_latency_ms, 0),
            'in_flight':   self._in_flight,
            'enabled':     self.enabled,
        }


# ─── SINGLETON ────────────────────────────────────────────────────────────────
GEMINI = GeminiEngine()
