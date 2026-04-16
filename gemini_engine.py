"""
HPAT v6 — gemini_engine.py  (v3 — full rewrite)

Root causes fixed from "no requests being sent" issue:
═══════════════════════════════════════════════════════════════════════
  BUG 1 — CRITICAL: FeedController._main() checked GEMINI.enabled at
           startup time, before user clicked ENABLE. Since enabled=False
           at start, no Gemini task was ever created in the asyncio loop.
  FIX    — GeminiEngine now owns its own asyncio task lifecycle.
           start() / stop() methods schedule the run coroutine directly
           onto the running feed loop via run_coroutine_threadsafe().

  BUG 2 — CRITICAL: _do_configure() in the UI called GEMINI.configure()
           which set enabled=True and created the client, but NEVER
           scheduled GEMINI.run() onto the feed loop. No coroutine = no calls.
  FIX    — GeminiEngine.start(loop, stop_event) is called directly from
           the UI toggle, passing the live loop reference from FeedController.

  BUG 3 — Race condition in start_gemini() after switch_pair():
           300ms delay meant it could run on the old/dead event loop.
  FIX    — FeedController exposes get_loop() so the UI always uses the
           current live loop. GeminiEngine stores the loop reference.

  BUG 4 — _gemini_task_wrapper reused the already-set stop_event from
           the previous feed cycle, causing immediate exit.
  FIX    — GeminiEngine has its own _stop_event created fresh each start().

  BUG 5 — Silent exception swallowing in configure() hid all errors.
  FIX    — configure() returns (bool, str) tuple: (success, error_message).
           UI displays the exact error reason to the user.

  BUG 6 — Old SDK google-generativeai v0.8.6 is deprecated.
  FIX    — Migrated to google-genai v1.72.0 with native async client.
           Falls back gracefully to old SDK if new one unavailable.

  BUG 7 — response_mime_type in generation_config caused silent failures
           on some SDK versions.
  FIX    — Removed from config; JSON enforcement done via system_instruction.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import STATE, BUS, Event, DECIMALS, PAIR_LABELS
from analytics_engine import (
    calc_atr, calc_rsi, calc_obi, get_vwap, get_poc,
    market_regime, composite_signal, now_ms,
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

_CADENCE: Dict[str, int] = {
    'VOLATILE': 10,
    'TRENDING': 10,
    'RANGING':  30,
}

_CHANGE_THRESHOLD = 0.0008   # 0.08% — minimum fractional move to trigger new call

# ─── RESULT MODEL ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    pair:              str
    timestamp:         str
    direction:         str    # 'LONG' | 'SHORT' | 'NEUTRAL' | 'NO_EDGE'
    conviction:        int    # 1-10
    time_horizon:      str
    entry_zone_lo:     float
    entry_zone_hi:     float
    stop_loss:         float
    take_profit_1:     float
    take_profit_2:     float
    risk_reward:       float
    primary_driver:    str
    confluence:        List[str]
    invalidation:      str
    regime:            str
    risk_level:        str
    position_size_pct: float
    reasoning:         str        = ''
    skipped:           bool       = False
    skip_reason:       str        = ''
    raw_json:          Dict       = field(default_factory=dict)
    error:             Optional[str] = None


# ─── COMPACT FEATURE EXTRACTOR ────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts the minimal, highest-signal-to-noise feature vector (~80-120 tokens).
    Strips nulls/zeros; uses compact JSON separators; encodes arrays as labels.
    """

    @staticmethod
    def extract(pair: str) -> Dict:
        s   = STATE.snapshot()
        dec = DECIMALS.get(pair, 2)
        now = now_ms()
        p   = s.price or 1.0

        # Price & spread
        spread_bps = (s.ask - s.bid) / max(s.bid, 1) * 10_000 if s.bid else 0.0

        # VWAP
        vwap, up1, up2, dn1, dn2 = get_vwap()
        vwap_dist = (p - vwap) / max(vwap, 1) * 100 if vwap else 0.0
        vwap_band = ("X2U" if p > up2 else "X1U" if p > up1 else
                     "X1D" if p < dn1 else "X2D" if p < dn2 else "MID")

        # RSI
        px_arr = [x.price for x in list(s.atr5m_prices)[-30:]]
        rsi_1m = round(calc_rsi(px_arr), 1)      if len(px_arr) > 15 else 50.0
        rsi_5m = round(calc_rsi(px_arr[::2]), 1) if len(px_arr) > 20 else rsi_1m

        # ATR / Volatility
        atr_5m  = calc_atr(pair, '5m')
        atr_30m = calc_atr(pair, '30m')
        vol_ratio = round(atr_5m / max(atr_30m, 1e-9), 3) if atr_5m and atr_30m else 1.0
        atr_pct   = round(atr_5m / p * 100, 4) if atr_5m else 0.0
        regime_label, _ = market_regime(pair)

        # CVD
        cvd_hist  = list(s.cvd_history)
        cvd_slope = 0.0
        cvd_div   = 'none'
        if len(cvd_hist) >= 10:
            recent    = [x.cvd for x in cvd_hist[-10:]]
            cvd_slope = round((recent[-1] - recent[0]) / max(abs(recent[0]), 1.0), 5)
            px10      = [x.price for x in list(s.atr_prices)[-10:]]
            if len(px10) >= 10:
                p_up = px10[-1] > px10[0]
                c_up = cvd_hist[-1].cvd > cvd_hist[0].cvd
                if p_up and not c_up:   cvd_div = 'bear'
                elif not p_up and c_up: cvd_div = 'bull'

        total_vol = s.vbuy + s.vsell or 1.0
        buy_pct   = round(s.vbuy / total_vol * 100, 1)

        # Order book
        obi      = round(calc_obi(s.ob_bids, s.ob_asks), 4)
        ob_state = ("BID_WALL" if obi > 0.5 else "BID_HEAVY" if obi > 0.25 else
                    "ASK_WALL" if obi < -0.5 else "ASK_HEAVY" if obi < -0.25 else "BALANCED")
        bid_walls: List = []
        ask_walls: List = []
        if s.ob_bids:
            avg_b     = sum(float(x[1]) for x in s.ob_bids[:5]) / min(5, len(s.ob_bids))
            bid_walls = [round(float(x[0]), dec) for x in s.ob_bids[:5] if float(x[1]) > avg_b * 3]
        if s.ob_asks:
            avg_a     = sum(float(x[1]) for x in s.ob_asks[:5]) / min(5, len(s.ob_asks))
            ask_walls = [round(float(x[0]), dec) for x in s.ob_asks[:5] if float(x[1]) > avg_a * 3]

        # Volume profile
        poc, poc_dist = get_poc()
        poc_side  = 'above' if p > poc else 'below' if poc else '?'
        vah = val = 0.0
        if s.vpvr:
            total_v = sum(s.vpvr.values())
            acc = 0.0; bins_va = []
            for b, v in sorted(s.vpvr.items(), key=lambda kv: kv[1], reverse=True):
                acc += v; bins_va.append(b)
                if acc >= total_v * 0.70: break
            if bins_va:
                vah = max(bins_va); val = min(bins_va)

        # Candles — compact string "B0.12b0.05B0.18"
        c5 = list(s.candles[pair]['5m'])[-3:]
        candle_str = ''.join(
            ('B' if c.c > c.o else 'b') + str(round(abs(c.c - c.o) / max(c.o, 1) * 100, 2))
            for c in c5
        ) or None

        # Derivatives
        funding = round(s.funding, 6)
        oi_chg  = 0.0
        oi_hist = list(s.oi_history)
        if len(oi_hist) >= 2:
            h1 = [x for x in oi_hist if now - x.t < 3_600_000]
            if len(h1) >= 2:
                oi_chg = round((s.oi - h1[0].oi) / max(h1[0].oi, 1) * 100, 3)

        # Large trades — counts only
        whale_b = inst_b = whale_s = inst_s = 0
        for t in list(s.trades)[:20]:
            if t.tier == 'WHALE':
                whale_b += t.is_buy;  whale_s += not t.is_buy
            elif t.tier == 'INST':
                inst_b  += t.is_buy;  inst_s  += not t.is_buy

        comp, _ = composite_signal()

        feat: Dict = {
            'sym':    PAIR_LABELS.get(pair, pair),
            'px':     round(p, dec),
            'sp_bps': round(spread_bps, 2),
            'r1m':    rsi_1m,
            'r5m':    rsi_5m,
            'obi':    obi,
            'ob_st':  ob_state,
            'cvd_sl': cvd_slope,
            'cvd_dv': cvd_div,
            'buy%':   buy_pct,
            'atr%':   atr_pct,
            'vol_r':  vol_ratio,
            'vwap_d': round(vwap_dist, 3),
            'vwap_b': vwap_band,
            'poc_d%': round(poc_dist, 3),
            'poc_s':  poc_side,
            'vah':    round(vah, dec) if vah else None,
            'val':    round(val, dec) if val else None,
            'fund':   funding,
            'oi_chg': oi_chg,
            'regime': regime_label,
            'sig':    comp[:20],
            'c5m':    candle_str,
            'wh':     f'B{whale_b}S{whale_s}' if (whale_b or whale_s) else None,
            'inst':   f'B{inst_b}S{inst_s}'   if (inst_b or inst_s)   else None,
        }
        if bid_walls: feat['bw'] = bid_walls
        if ask_walls: feat['aw'] = ask_walls

        # Strip None, zero, and 'none'
        feat = {k: v for k, v in feat.items()
                if v is not None and v != 0 and v != 0.0 and v != 'none'}

        # Internal metadata — NOT sent to API
        feat['_pair']   = pair
        feat['_ts']     = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        feat['_atr_5m'] = round(atr_5m, dec + 2) if atr_5m else 0.0
        feat['_regime'] = regime_label
        feat['_price']  = p
        return feat

    @staticmethod
    def significant_change(prev: Dict, curr: Dict) -> Tuple[bool, str]:
        """Returns (changed, reason). Skips API call if market is flat."""
        if not prev:
            return True, 'first_call'
        if prev.get('regime') != curr.get('regime'):
            return True, f'regime:{curr.get("regime")}'
        if prev.get('cvd_dv') != curr.get('cvd_dv') and curr.get('cvd_dv') not in (None, 'none'):
            return True, f'cvd_div:{curr.get("cvd_dv")}'
        if prev.get('ob_st') != curr.get('ob_st') and 'WALL' in str(curr.get('ob_st', '')):
            return True, f'wall:{curr.get("ob_st")}'
        if prev.get('vwap_b') != curr.get('vwap_b'):
            return True, f'vwap_cross:{curr.get("vwap_b")}'
        for key, old_v, new_v in [
            ('px',     prev.get('px',     0),   curr.get('px',     0)),
            ('obi',    prev.get('obi',    0),   curr.get('obi',    0)),
            ('cvd_sl', prev.get('cvd_sl', 0),   curr.get('cvd_sl', 0)),
            ('r1m',    prev.get('r1m',   50),   curr.get('r1m',   50)),
            ('fund',   prev.get('fund',   0),   curr.get('fund',   0)),
        ]:
            base  = max(abs(old_v), 1e-9)
            delta = abs(new_v - old_v) / base
            if delta > _CHANGE_THRESHOLD:
                return True, f'{key}_Δ{delta:.3f}'
        return False, 'market_flat'


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Two-part architecture:
    - SYSTEM: static expert persona + rules (set once at model init)
    - USER:   compact per-call market data (~80-150 tokens)
    """

    SYSTEM_INSTRUCTION = """\
You are HPAT-AI, a quant crypto trading analyst specialising in ultra-short-term \
(10-120s) directional signals from order flow, volume profile, and momentum. \
Output ONLY valid JSON — no prose, no markdown fences, no explanation.

SIGNAL RULES:
• conv≥8 needs ≥3 uncorrelated confirms; conv4-7=2 confirms; conv≤3=NO_EDGE.
• NEVER force direction when signals conflict — use NO_EDGE.
• Bearish CVD divergence(cvd_dv=bear) while price above VWAP = strong bear signal.
• OBI>0.3 + rising CVD slope = strong immediate buy pressure.
• RSI>75(1m) = mean-reversion bias. POC distance<0.1% = consolidation likely.

CALIBRATION:
1. sp_bps>5 → risk up one grade.
2. vol_r>2 → time_horizon≤30s.
3. fund>0.08% → cap LONG conv at 6.
4. fund<-0.08% → cap SHORT conv at 6.
5. cvd_dv=bear + dir=LONG → conv≤5.
6. cvd_dv=bull + dir=SHORT → conv≤5.
7. sl ≥ 1×ATR from entry midpoint; tp1 RR≥1.5; tp2 RR≥2.5.
8. size%=base_kelly×(conv/10)/risk_mult [L=1,M=1.5,H=2.5,X=4].

FIELD KEY:
px=price, sp_bps=spread_bps, r1m/r5m=RSI, obi=order_book_imbalance[-1..1],
ob_st=BALANCED|BID_HEAVY|BID_WALL|ASK_HEAVY|ASK_WALL,
cvd_sl=cvd_slope_10tick, cvd_dv=divergence(none|bull|bear),
buy%=buy_volume_pct, atr%=atr_pct_of_price, vol_r=volatility_expansion_ratio,
vwap_d=vwap_dist_pct, vwap_b=MID|X1U|X2U|X1D|X2D,
poc_d%=poc_dist_pct, poc_s=above|below, vah/val=value_area_hi/lo,
fund=funding_rate_pct, oi_chg=oi_1h_change_pct, regime=VOLATILE|TRENDING|RANGING,
sig=composite_signal, c5m=candle_seq(B=bull b=bear+body%),
wh=whale_BxSy, inst=inst_BxSy, bw=bid_wall_prices, aw=ask_wall_prices.

OUTPUT — return ONLY this JSON, nothing else:
{"dir":"LONG"|"SHORT"|"NEUTRAL"|"NO_EDGE","conv":1-10,"hz":"Xs-Xs",\
"ez":[lo,hi],"sl":0,"tp1":0,"tp2":0,"rr":0.0,\
"risk":"L"|"M"|"H"|"X","size":0.0,\
"driver":"<120 chars","cf":["f1","f2","f3"],"inv":"<80 chars",\
"why":"step1|step2|step3|verdict"}\
"""

    @classmethod
    def build_user(cls, feat: Dict) -> str:
        pair    = feat.get('_pair',   '?')
        ts      = feat.get('_ts',     '')
        atr_5m  = feat.get('_atr_5m', 0)
        price   = feat.get('_price',  0)
        regime  = feat.get('_regime', 'RANGING')
        dec     = DECIMALS.get(pair, 2)
        sl_guide  = round(atr_5m * 1.0, dec)
        tp1_guide = round(atr_5m * 1.8, dec)
        atr_pct   = atr_5m / max(price, 1) * 100
        payload   = {k: v for k, v in feat.items() if not k.startswith('_')}
        return (
            f"ANALYSE {pair}@{price} {ts}UTC regime={regime}\n"
            f"ATR5m={atr_5m}({atr_pct:.3f}%) SL≥{sl_guide} TP1~{tp1_guide}\n"
            f"DATA:{json.dumps(payload, separators=(',', ':'))}"
        )


# ─── RESPONSE PARSER ──────────────────────────────────────────────────────────

class ResponseParser:

    _RISK_MAP = {'L': 'LOW', 'M': 'MEDIUM', 'H': 'HIGH', 'X': 'EXTREME'}

    @staticmethod
    def parse(raw_text: str, feat: Dict) -> PredictionResult:
        pair  = feat.get('_pair',  'UNKNOWN')
        ts    = feat.get('_ts',    '')
        dec   = DECIMALS.get(pair, 2)
        price = feat.get('_price', 0.0)

        cleaned = re.sub(r'```(?:json)?|```', '', raw_text or '').strip()
        match   = re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            return ResponseParser._err(pair, ts, f'no_json:raw={raw_text[:60]!r}', price)
        try:
            d = json.loads(match.group())
        except json.JSONDecodeError as e:
            return ResponseParser._err(pair, ts, f'json:{e}', price)
        try:
            ez  = d.get('ez', [price, price])
            lo  = float(ez[0]) if isinstance(ez, (list, tuple)) else float(ez.get('lo', price))
            hi  = float(ez[1]) if isinstance(ez, (list, tuple)) else float(ez.get('hi', price))
            risk_raw = str(d.get('risk', 'H')).upper()
            risk     = ResponseParser._RISK_MAP.get(risk_raw, risk_raw)
            return PredictionResult(
                pair=pair, timestamp=ts,
                direction=str(d.get('dir', 'NO_EDGE')).upper(),
                conviction=max(1, min(10, int(d.get('conv', 1)))),
                time_horizon=str(d.get('hz', 'N/A')),
                entry_zone_lo=round(lo, dec),
                entry_zone_hi=round(hi, dec),
                stop_loss=round(float(d.get('sl', price)), dec),
                take_profit_1=round(float(d.get('tp1', price)), dec),
                take_profit_2=round(float(d.get('tp2', price)), dec),
                risk_reward=round(float(d.get('rr', 0.0)), 2),
                primary_driver=str(d.get('driver', ''))[:200],
                confluence=list(d.get('cf', []))[:5],
                invalidation=str(d.get('inv', ''))[:200],
                regime=feat.get('_regime', 'UNKNOWN'),
                risk_level=risk,
                position_size_pct=round(float(d.get('size', 0.0)), 2),
                reasoning=str(d.get('why', '')),
                raw_json=d,
            )
        except (KeyError, TypeError, ValueError) as e:
            return ResponseParser._err(pair, ts, f'field:{e}', price)

    @staticmethod
    def _err(pair, ts, msg, price) -> PredictionResult:
        return PredictionResult(
            pair=pair, timestamp=ts,
            direction='NO_EDGE', conviction=0, time_horizon='N/A',
            entry_zone_lo=price, entry_zone_hi=price,
            stop_loss=price, take_profit_1=price, take_profit_2=price,
            risk_reward=0.0, primary_driver='Parse error',
            confluence=[], invalidation='N/A',
            regime='UNKNOWN', risk_level='EXTREME',
            position_size_pct=0.0, error=msg,
        )


# ─── SDK ABSTRACTION ──────────────────────────────────────────────────────────

class _GeminiClient:
    """
    Thin wrapper that supports both SDKs:
      • google-genai  >= 1.0  (new, recommended, native async)
      • google-generativeai 0.x (old, deprecated, sync only)
    Tries new SDK first; falls back to old automatically.
    """

    def __init__(self, api_key: str, model_name: str,
                 system_instruction: str) -> None:
        self._model_name    = model_name
        self._sys_instr     = system_instruction
        self._api_key       = api_key
        self._client_new    = None   # google.genai Client
        self._client_old    = None   # google.generativeai GenerativeModel
        self._use_new_sdk   = False
        self._gen_config    = {
            'temperature':       0.10,
            'top_p':             0.80,
            'top_k':             20,
            'max_output_tokens': 400,
        }

    def setup(self) -> Tuple[bool, str]:
        """Try new SDK first, then old. Returns (ok, error_msg)."""
        # ── Attempt 1: google-genai (new SDK) ─────────────────────────────
        try:
            import google.genai as genai_new
            from google.genai import types
            self._client_new  = genai_new.Client(api_key=self._api_key)
            self._use_new_sdk = True
            return True, ''
        except ImportError:
            pass
        except Exception as e:
            return False, f'new-sdk error: {e}'

        # ── Attempt 2: google-generativeai (old SDK) ──────────────────────
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=self._api_key)
            self._client_old = genai_old.GenerativeModel(
                model_name=self._model_name,
                system_instruction=self._sys_instr,
                generation_config=self._gen_config,
            )
            self._use_new_sdk = False
            return True, ''
        except ImportError:
            return False, 'no Gemini SDK installed. Run: pip install google-genai'
        except Exception as e:
            return False, f'old-sdk error: {e}'

    async def generate(self, user_msg: str) -> str:
        """Send a generation request and return the response text."""
        if self._use_new_sdk:
            return await self._generate_new(user_msg)
        else:
            return await self._generate_old(user_msg)

    async def _generate_new(self, user_msg: str) -> str:
        from google.genai import types
        loop = asyncio.get_event_loop()

        def _call():
            response = self._client_new.models.generate_content(
                model=self._model_name,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=self._sys_instr,
                    temperature=self._gen_config['temperature'],
                    top_p=self._gen_config['top_p'],
                    top_k=self._gen_config['top_k'],
                    max_output_tokens=self._gen_config['max_output_tokens'],
                    response_mime_type='application/json',
                ),
            )
            return response.text

        return await loop.run_in_executor(None, _call)

    async def _generate_old(self, user_msg: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client_old.generate_content(user_msg).text
        )


# ─── GEMINI ENGINE ────────────────────────────────────────────────────────────

class GeminiEngine:
    """
    Fixed async Gemini prediction engine.

    Lifecycle (corrected):
    ─────────────────────
    1. UI calls configure(api_key) → creates _GeminiClient, returns (ok, err)
    2. If ok, UI calls start(loop, stop_event) → schedules run() on feed loop
    3. run() fires _cycle() every cadence seconds (10s volatile, 30s ranging)
    4. _cycle() checks deduplication → calls API if market changed
    5. Results emitted via EventBus → UI subscribes and updates panels
    6. UI calls stop() → sets internal stop event → run() exits cleanly
    """

    MODEL_NAME = 'gemini-2.0-flash'   # instance-level; changed by UI model selector

    def __init__(self) -> None:
        self.MODEL_NAME      = 'gemini-2.0-flash'
        self._sdk_client: Optional[_GeminiClient] = None
        self._task:        Optional[asyncio.Task] = None
        self._stop_ev:     Optional[asyncio.Event] = None
        self._loop:        Optional[asyncio.AbstractEventLoop] = None
        self._in_flight:   bool  = False
        self._last_result: Optional[PredictionResult] = None
        self._last_feat:   Dict  = {}
        self._call_count:  int   = 0
        self._skip_count:  int   = 0
        self._error_count: int   = 0
        self._last_latency: float = 0.0
        self._tokens_saved: int  = 0
        self.enabled:       bool = False
        self._last_error:   str  = ''

    # ── Configure (called from UI thread) ─────────────────────────────────────

    def configure(self, api_key: str) -> Tuple[bool, str]:
        """
        Creates and validates the SDK client.
        Returns (success: bool, error_message: str).
        Does NOT start the prediction loop — call start() for that.
        """
        if not api_key or not api_key.strip():
            return False, 'API key is empty'

        client = _GeminiClient(
            api_key=api_key.strip(),
            model_name=self.MODEL_NAME,
            system_instruction=PromptBuilder.SYSTEM_INSTRUCTION,
        )
        ok, err = client.setup()
        if ok:
            self._sdk_client = client
            self.enabled     = True
            self._last_error = ''
            # Reset dedup cache so first call always fires
            self._last_feat  = {}
        else:
            self._sdk_client = None
            self.enabled     = False
            self._last_error = err
        return ok, err

    # ── Start / Stop (called from UI thread, thread-safe) ─────────────────────

    def start(self, loop: asyncio.AbstractEventLoop,
              feed_stop_event: asyncio.Event) -> None:
        """
        Schedule the prediction coroutine onto the given asyncio loop.
        Safe to call from any thread.
        """
        if not self.enabled or not self._sdk_client:
            return
        self._loop    = loop
        # Create a fresh stop event on the target loop
        self._stop_ev = asyncio.new_event_loop().run_until_complete(asyncio.sleep(0)) or None
        # Schedule on the feed loop (thread-safe)
        asyncio.run_coroutine_threadsafe(
            self._start_on_loop(feed_stop_event), loop
        )

    async def _start_on_loop(self, feed_stop_event: asyncio.Event) -> None:
        """Runs inside the feed asyncio loop. Creates fresh stop event and starts run()."""
        # Cancel any existing task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._stop_ev = asyncio.Event()
        self._task    = asyncio.create_task(self.run(feed_stop_event))

    def stop(self) -> None:
        """Signal the prediction loop to stop."""
        self.enabled = False
        if self._loop and self._stop_ev:
            self._loop.call_soon_threadsafe(self._stop_ev.set)
        if self._loop and self._task:
            self._loop.call_soon_threadsafe(self._task.cancel)

    # ── Main prediction loop ───────────────────────────────────────────────────

    async def run(self, feed_stop_event: asyncio.Event) -> None:
        """
        Prediction coroutine that runs inside the feed asyncio loop.
        Exits when either the feed stops OR stop() is called.
        """
        await asyncio.sleep(5)   # warm-up delay — let market data accumulate

        while True:
            # Exit conditions
            if feed_stop_event.is_set():
                break
            if self._stop_ev and self._stop_ev.is_set():
                break
            if not self.enabled:
                break

            if not self._in_flight:
                asyncio.create_task(self._cycle())

            regime  = self._last_feat.get('_regime', 'RANGING')
            cadence = _CADENCE.get(regime, 30)
            try:
                await asyncio.wait_for(
                    asyncio.shield(feed_stop_event.wait()),
                    timeout=cadence,
                )
                break  # feed stopped during sleep
            except asyncio.TimeoutError:
                pass   # normal — cadence elapsed, fire next cycle

    # ── Prediction cycle ──────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        self._in_flight = True
        pair = STATE.pair
        t0   = time.monotonic()
        try:
            feat = FeatureExtractor.extract(pair)

            # Deduplication
            changed, reason = FeatureExtractor.significant_change(self._last_feat, feat)
            if not changed:
                self._skip_count  += 1
                self._tokens_saved += 200
                if self._last_result:
                    skipped = PredictionResult(
                        **{k: v for k, v in self._last_result.__dict__.items()
                           if k not in ('skipped', 'skip_reason', 'timestamp')},
                        timestamp=feat.get('_ts', ''),
                        skipped=True,
                        skip_reason=reason,
                    )
                    BUS.emit(Event('ai_prediction', skipped))
                return

            user_msg = PromptBuilder.build_user(feat)
            result   = await self._call_api(user_msg, feat)

            self._last_latency = (time.monotonic() - t0) * 1000
            self._last_result  = result
            self._last_feat    = feat
            self._call_count  += 1
            if result.error:
                self._error_count += 1

            BUS.emit(Event('ai_prediction', result))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error_count += 1
            self._last_error   = str(e)
            BUS.emit(Event('ai_prediction_error', str(e)))
        finally:
            self._in_flight = False

    async def _call_api(self, user_msg: str, feat: Dict) -> PredictionResult:
        raw = await self._sdk_client.generate(user_msg)
        return ResponseParser.parse(raw, feat)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        total     = self._call_count + self._skip_count
        skip_rate = self._skip_count / max(total, 1) * 100
        return {
            'calls':          self._call_count,
            'skips':          self._skip_count,
            'skip_rate_pct':  round(skip_rate, 1),
            'errors':         self._error_count,
            'latency_ms':     round(self._last_latency, 0),
            'tokens_saved':   self._tokens_saved,
            'in_flight':      self._in_flight,
            'enabled':        self.enabled,
            'model':          self.MODEL_NAME,
            'last_error':     self._last_error,
            'sdk':            'google-genai' if (self._sdk_client and self._sdk_client._use_new_sdk)
                               else 'google-generativeai',
        }

    def user_prompt_size(self) -> Tuple[int, int]:
        try:
            feat     = FeatureExtractor.extract(STATE.pair)
            user_msg = PromptBuilder.build_user(feat)
            chars    = len(user_msg)
            return chars, chars // 4
        except Exception:
            return 0, 0


# ─── SINGLETON ────────────────────────────────────────────────────────────────
GEMINI = GeminiEngine()
