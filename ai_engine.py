"""
HPAT v6 — ai_engine.py
Multi-provider AI prediction engine.

Supported providers:
  ┌──────────────┬────────────────────────────────────────────────────────┐
  │ GEMINI       │ google-genai SDK + google-generativeai fallback        │
  │              │ Models: gemini-2.0-flash, flash-lite, 1.5-flash, 1.5-pro│
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ GROQ         │ groq SDK (native async) — fastest inference available  │
  │              │ Models: llama-3.3-70b, llama-3.1-70b, mixtral-8x7b,   │
  │              │         gemma2-9b, llama-3.1-8b                        │
  ├──────────────┼────────────────────────────────────────────────────────┤
  │ OPENROUTER   │ OpenAI-compatible API (openai SDK, custom base_url)    │
  │              │ Routes to 100+ models: GPT-4o, Claude 3.5, Llama etc.  │
  └──────────────┴────────────────────────────────────────────────────────┘

Architecture:
  • ProviderClient   — abstract interface: setup() + generate()
  • GeminiClient     — wraps google-genai (new) + google-generativeai (fallback)
  • GroqClient       — wraps groq.AsyncGroq with JSON-mode enforcement
  • OpenRouterClient — wraps openai.AsyncOpenAI with OR base URL
  • AIEngine         — provider-agnostic engine (configure/start/stop/lifecycle)
  • Singleton AI     — imported and used everywhere as `from ai_engine import AI`

All providers share:
  • Same FeatureExtractor (compact 80-120 token feature payload)
  • Same PromptBuilder (system instruction + per-call user turn)
  • Same ResponseParser (compact JSON → PredictionResult)
  • Same deduplication, adaptive cadence, and async lifecycle
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import STATE, BUS, Event, DECIMALS, PAIR_LABELS
from analytics_engine import (
    calc_atr, calc_rsi, calc_obi, get_vwap, get_poc,
    market_regime, composite_signal, now_ms,
)

# ─── PROVIDER REGISTRY ────────────────────────────────────────────────────────

PROVIDERS = {
    'gemini': {
        'name':    'Google Gemini',
        'color':   '#00eeff',
        'env_key': 'GEMINI_API_KEY',
        'models': [
            {'id': 'gemini-2.0-flash',      'label': 'Gemini 2.0 Flash',      'badge': '2.0 FLASH', 'tier': 'FAST',    'desc': 'Best speed/accuracy. Recommended for real-time.'},
            {'id': 'gemini-2.0-flash-lite', 'label': 'Gemini 2.0 Flash-Lite', 'badge': '2.0 LITE',  'tier': 'ECONOMY', 'desc': 'Fastest & cheapest. Great for high-frequency.'},
            {'id': 'gemini-1.5-flash',      'label': 'Gemini 1.5 Flash',      'badge': '1.5 FLASH', 'tier': 'STABLE',  'desc': 'Proven stable model. Good for production.'},
            {'id': 'gemini-1.5-pro',        'label': 'Gemini 1.5 Pro',        'badge': '1.5 PRO',   'tier': 'DEEP',    'desc': 'Deepest reasoning. Complex multi-factor analysis.'},
        ],
    },
    'groq': {
        'name':    'Groq Cloud',
        'color':   '#ff6b35',
        'env_key': 'GROQ_API_KEY',
        'models': [
            {'id': 'llama-3.3-70b-versatile',      'label': 'Llama 3.3 70B',    'badge': 'LLaMA 3.3', 'tier': 'SMART',   'desc': 'Best Groq model. Deep reasoning + ultra-fast.'},
            {'id': 'llama-3.1-70b-versatile',      'label': 'Llama 3.1 70B',    'badge': 'LLaMA 3.1', 'tier': 'FAST',    'desc': 'Great balance of speed and intelligence.'},
            {'id': 'mixtral-8x7b-32768',           'label': 'Mixtral 8x7B',     'badge': 'MoE 8x7B',  'tier': 'FAST',    'desc': 'Mixture-of-experts. Fast & capable.'},
            {'id': 'gemma2-9b-it',                 'label': 'Gemma 2 9B',       'badge': 'GEMMA2',    'tier': 'LEAN',    'desc': 'Google Gemma 2. Efficient instruction model.'},
            {'id': 'llama-3.1-8b-instant',         'label': 'Llama 3.1 8B',     'badge': 'LLaMA 8B',  'tier': 'TURBO',   'desc': 'Ultra-fast. Best latency of all options.'},
        ],
    },
    'openrouter': {
        'name':    'OpenRouter',
        'color':   '#7c5cfc',
        'env_key': 'OPENROUTER_API_KEY',
        'models': [
            {'id': 'openai/gpt-4o-mini',            'label': 'GPT-4o Mini',          'badge': 'GPT4o-M',  'tier': 'FAST',   'desc': 'OpenAI GPT-4o Mini. Fast & affordable.'},
            {'id': 'openai/gpt-4o',                 'label': 'GPT-4o',               'badge': 'GPT4o',    'tier': 'SMART',  'desc': 'Full GPT-4o. Best OpenAI analysis quality.'},
            {'id': 'anthropic/claude-3.5-haiku',    'label': 'Claude 3.5 Haiku',     'badge': 'HAIKU',    'tier': 'FAST',   'desc': 'Anthropic Haiku. Fast, precise analysis.'},
            {'id': 'anthropic/claude-3.5-sonnet',   'label': 'Claude 3.5 Sonnet',    'badge': 'SONNET',   'tier': 'DEEP',   'desc': 'Anthropic Sonnet. Best reasoning quality.'},
            {'id': 'meta-llama/llama-3.3-70b-instruct', 'label': 'Llama 3.3 70B',   'badge': 'LLaMA',    'tier': 'SMART',  'desc': 'Meta Llama 3.3 via OpenRouter. Free tier.'},
            {'id': 'google/gemini-2.0-flash-exp:free', 'label': 'Gemini 2.0 Flash Free', 'badge': 'FREE', 'tier': 'FREE',   'desc': 'Gemini 2.0 Flash — free tier via OpenRouter.'},
        ],
    },
}

# Flat model catalogue: id → {provider, ...model_info}
ALL_MODELS: Dict[str, Dict] = {}
for _prov_id, _prov in PROVIDERS.items():
    for _m in _prov['models']:
        ALL_MODELS[_m['id']] = {**_m, 'provider': _prov_id,
                                  'provider_name': _prov['name'],
                                  'provider_color': _prov['color']}

# ─── ADAPTIVE CADENCE ─────────────────────────────────────────────────────────

_CADENCE: Dict[str, int] = {'VOLATILE': 10, 'TRENDING': 10, 'RANGING': 30}
_CHANGE_THRESHOLD = 0.0008   # 0.08% minimum fractional move

# ─── RESULT MODEL ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    pair:              str
    timestamp:         str
    provider:          str   # 'gemini' | 'groq' | 'openrouter'
    model_id:          str
    direction:         str   # 'LONG' | 'SHORT' | 'NEUTRAL' | 'NO_EDGE'
    conviction:        int   # 1-10
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


# ─── FEATURE EXTRACTOR ────────────────────────────────────────────────────────

class FeatureExtractor:
    """Compact ~80-120 token feature vector. Shared by all providers."""

    @staticmethod
    def extract(pair: str) -> Dict:
        s   = STATE.snapshot()
        dec = DECIMALS.get(pair, 2)
        now = now_ms()
        p   = s.price or 1.0

        spread_bps = (s.ask - s.bid) / max(s.bid, 1) * 10_000 if s.bid else 0.0

        vwap, up1, up2, dn1, dn2 = get_vwap()
        vwap_dist = (p - vwap) / max(vwap, 1) * 100 if vwap else 0.0
        vwap_band = ("X2U" if p > up2 else "X1U" if p > up1 else
                     "X1D" if p < dn1 else "X2D" if p < dn2 else "MID")

        px_arr = [x.price for x in list(s.atr5m_prices)[-30:]]
        rsi_1m = round(calc_rsi(px_arr), 1)      if len(px_arr) > 15 else 50.0
        rsi_5m = round(calc_rsi(px_arr[::2]), 1) if len(px_arr) > 20 else rsi_1m

        atr_5m  = calc_atr(pair, '5m')
        atr_30m = calc_atr(pair, '30m')
        vol_ratio = round(atr_5m / max(atr_30m, 1e-9), 3) if atr_5m and atr_30m else 1.0
        atr_pct   = round(atr_5m / p * 100, 4) if atr_5m else 0.0
        regime_label, _ = market_regime(pair)

        cvd_hist  = list(s.cvd_history)
        cvd_slope = 0.0
        cvd_div   = 'none'
        if len(cvd_hist) >= 10:
            recent    = [x.cvd for x in cvd_hist[-10:]]
            cvd_slope = round((recent[-1] - recent[0]) / max(abs(recent[0]), 1.0), 5)
            px10      = [x.price for x in list(s.atr_prices)[-10:]]
            if len(px10) >= 10:
                p_up = px10[-1] > px10[0];  c_up = cvd_hist[-1].cvd > cvd_hist[0].cvd
                if p_up and not c_up:    cvd_div = 'bear'
                elif not p_up and c_up:  cvd_div = 'bull'

        total_vol = s.vbuy + s.vsell or 1.0
        buy_pct   = round(s.vbuy / total_vol * 100, 1)

        obi      = round(calc_obi(s.ob_bids, s.ob_asks), 4)
        ob_state = ("BID_WALL" if obi > 0.5 else "BID_HEAVY" if obi > 0.25 else
                    "ASK_WALL" if obi < -0.5 else "ASK_HEAVY" if obi < -0.25 else "BALANCED")
        bid_walls: List = []
        ask_walls: List = []
        if s.ob_bids:
            avg_b = sum(float(x[1]) for x in s.ob_bids[:5]) / min(5, len(s.ob_bids))
            bid_walls = [round(float(x[0]), dec) for x in s.ob_bids[:5] if float(x[1]) > avg_b * 3]
        if s.ob_asks:
            avg_a = sum(float(x[1]) for x in s.ob_asks[:5]) / min(5, len(s.ob_asks))
            ask_walls = [round(float(x[0]), dec) for x in s.ob_asks[:5] if float(x[1]) > avg_a * 3]

        poc, poc_dist = get_poc()
        poc_side = 'above' if p > poc else 'below' if poc else '?'
        vah = val = 0.0
        if s.vpvr:
            total_v = sum(s.vpvr.values()); acc = 0.0; bins_va = []
            for b, v in sorted(s.vpvr.items(), key=lambda kv: kv[1], reverse=True):
                acc += v; bins_va.append(b)
                if acc >= total_v * 0.70: break
            if bins_va: vah = max(bins_va); val = min(bins_va)

        c5 = list(s.candles[pair]['5m'])[-3:]
        candle_str = ''.join(
            ('B' if c.c > c.o else 'b') + str(round(abs(c.c - c.o) / max(c.o, 1) * 100, 2))
            for c in c5
        ) or None

        funding = round(s.funding, 6)
        oi_chg  = 0.0
        oi_hist = list(s.oi_history)
        if len(oi_hist) >= 2:
            h1 = [x for x in oi_hist if now - x.t < 3_600_000]
            if len(h1) >= 2:
                oi_chg = round((s.oi - h1[0].oi) / max(h1[0].oi, 1) * 100, 3)

        whale_b = inst_b = whale_s = inst_s = 0
        for t in list(s.trades)[:20]:
            if t.tier == 'WHALE':   whale_b += t.is_buy;  whale_s += not t.is_buy
            elif t.tier == 'INST':  inst_b  += t.is_buy;  inst_s  += not t.is_buy

        comp, _ = composite_signal()

        feat: Dict = {
            'sym': PAIR_LABELS.get(pair, pair), 'px': round(p, dec),
            'sp_bps': round(spread_bps, 2),
            'r1m': rsi_1m, 'r5m': rsi_5m,
            'obi': obi, 'ob_st': ob_state,
            'cvd_sl': cvd_slope, 'cvd_dv': cvd_div, 'buy%': buy_pct,
            'atr%': atr_pct, 'vol_r': vol_ratio,
            'vwap_d': round(vwap_dist, 3), 'vwap_b': vwap_band,
            'poc_d%': round(poc_dist, 3), 'poc_s': poc_side,
            'vah': round(vah, dec) if vah else None,
            'val': round(val, dec) if val else None,
            'fund': funding, 'oi_chg': oi_chg,
            'regime': regime_label, 'sig': comp[:20], 'c5m': candle_str,
            'wh':  f'B{whale_b}S{whale_s}' if (whale_b or whale_s) else None,
            'inst': f'B{inst_b}S{inst_s}'  if (inst_b or inst_s)   else None,
        }
        if bid_walls: feat['bw'] = bid_walls
        if ask_walls: feat['aw'] = ask_walls
        feat = {k: v for k, v in feat.items()
                if v is not None and v != 0 and v != 0.0 and v != 'none'}

        feat['_pair']   = pair
        feat['_ts']     = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        feat['_atr_5m'] = round(atr_5m, dec + 2) if atr_5m else 0.0
        feat['_regime'] = regime_label
        feat['_price']  = p
        return feat

    @staticmethod
    def significant_change(prev: Dict, curr: Dict) -> Tuple[bool, str]:
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
            if abs(new_v - old_v) / base > _CHANGE_THRESHOLD:
                return True, f'{key}_Δ{abs(new_v - old_v)/base:.3f}'
        return False, 'market_flat'


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Universal prompt builder — works for all providers.
    System instruction is separated so providers that support it (Gemini, OpenRouter)
    can pass it as a separate field; others (Groq) get it prepended to the user turn.
    """

    SYSTEM_INSTRUCTION = """\
You are HPAT-AI, a quant crypto trading analyst specialising in ultra-short-term \
(10-120s) directional signals from order flow, volume profile, and momentum. \
Output ONLY valid JSON — no prose, no markdown fences, no explanation.

SIGNAL RULES:
• conv≥8 needs ≥3 uncorrelated confirms; conv4-7=2 confirms; conv≤3=NO_EDGE.
• NEVER force direction when signals conflict — use NO_EDGE.
• Bearish CVD divergence(cvd_dv=bear) + price above VWAP = strong bear signal.
• OBI>0.3 + rising CVD slope = strong buy pressure.
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
px=price sp_bps=spread_bps r1m/r5m=RSI obi=order_book_imbalance[-1..1]
ob_st=BALANCED|BID_HEAVY|BID_WALL|ASK_HEAVY|ASK_WALL
cvd_sl=cvd_slope_10tick cvd_dv=divergence(none|bull|bear)
buy%=buy_volume_pct atr%=atr_pct_of_price vol_r=volatility_expansion_ratio
vwap_d=vwap_dist_pct vwap_b=MID|X1U|X2U|X1D|X2D
poc_d%=poc_dist_pct poc_s=above|below vah/val=value_area_hi/lo
fund=funding_rate_pct oi_chg=oi_1h_change_pct regime=VOLATILE|TRENDING|RANGING
sig=composite_signal c5m=candle_seq(B=bull b=bear+body%) wh=whale_BxSy
inst=inst_BxSy bw=bid_wall_prices aw=ask_wall_prices

OUTPUT — return ONLY this JSON object, nothing else:
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
        sl_g    = round(atr_5m * 1.0, dec)
        tp1_g   = round(atr_5m * 1.8, dec)
        atr_pct = atr_5m / max(price, 1) * 100
        payload = {k: v for k, v in feat.items() if not k.startswith('_')}
        return (
            f"ANALYSE {pair}@{price} {ts}UTC regime={regime}\n"
            f"ATR5m={atr_5m}({atr_pct:.3f}%) SL≥{sl_g} TP1~{tp1_g}\n"
            f"DATA:{json.dumps(payload, separators=(',', ':'))}"
        )


# ─── RESPONSE PARSER ──────────────────────────────────────────────────────────

class ResponseParser:

    _RISK_MAP = {'L': 'LOW', 'M': 'MEDIUM', 'H': 'HIGH', 'X': 'EXTREME'}

    @staticmethod
    def parse(raw_text: str, feat: Dict,
              provider: str = 'unknown', model_id: str = '') -> PredictionResult:
        pair  = feat.get('_pair',  'UNKNOWN')
        ts    = feat.get('_ts',    '')
        dec   = DECIMALS.get(pair, 2)
        price = feat.get('_price', 0.0)

        cleaned = re.sub(r'```(?:json)?|```', '', raw_text or '').strip()
        match   = re.search(r'\{[\s\S]*\}', cleaned)
        if not match:
            return ResponseParser._err(pair, ts, provider, model_id,
                                        f'no_json:raw={raw_text[:60]!r}', price)
        try:
            d = json.loads(match.group())
        except json.JSONDecodeError as e:
            return ResponseParser._err(pair, ts, provider, model_id, f'json:{e}', price)

        try:
            ez  = d.get('ez', [price, price])
            lo  = float(ez[0]) if isinstance(ez, (list, tuple)) else float(ez.get('lo', price))
            hi  = float(ez[1]) if isinstance(ez, (list, tuple)) else float(ez.get('hi', price))
            risk_raw = str(d.get('risk', 'H')).upper()
            risk     = ResponseParser._RISK_MAP.get(risk_raw, risk_raw)
            return PredictionResult(
                pair=pair, timestamp=ts, provider=provider, model_id=model_id,
                direction=str(d.get('dir', 'NO_EDGE')).upper(),
                conviction=max(1, min(10, int(d.get('conv', 1)))),
                time_horizon=str(d.get('hz', 'N/A')),
                entry_zone_lo=round(lo, dec), entry_zone_hi=round(hi, dec),
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
            return ResponseParser._err(pair, ts, provider, model_id, f'field:{e}', price)

    @staticmethod
    def _err(pair, ts, provider, model_id, msg, price) -> PredictionResult:
        return PredictionResult(
            pair=pair, timestamp=ts, provider=provider, model_id=model_id,
            direction='NO_EDGE', conviction=0, time_horizon='N/A',
            entry_zone_lo=price, entry_zone_hi=price,
            stop_loss=price, take_profit_1=price, take_profit_2=price,
            risk_reward=0.0, primary_driver='Parse error',
            confluence=[], invalidation='N/A',
            regime='UNKNOWN', risk_level='EXTREME',
            position_size_pct=0.0, error=msg,
        )


# ─── PROVIDER CLIENTS ─────────────────────────────────────────────────────────

class ProviderClient(ABC):
    """Abstract base for all provider clients."""

    def __init__(self, api_key: str, model_id: str) -> None:
        self.api_key  = api_key.strip()
        self.model_id = model_id

    @abstractmethod
    def setup(self) -> Tuple[bool, str]:
        """Validate configuration. Returns (ok, error_message)."""

    @abstractmethod
    async def generate(self, user_msg: str) -> str:
        """Send generation request, return raw response text."""

    @property
    def provider_id(self) -> str:
        return 'unknown'


class GeminiClient(ProviderClient):
    """Supports google-genai (new) + google-generativeai (fallback)."""

    def __init__(self, api_key: str, model_id: str) -> None:
        super().__init__(api_key, model_id)
        self._client_new = None
        self._client_old = None
        self._use_new    = False
        self._gen_cfg    = {'temperature': 0.10, 'top_p': 0.80,
                            'top_k': 20, 'max_output_tokens': 400}

    @property
    def provider_id(self) -> str:
        return 'gemini'

    def setup(self) -> Tuple[bool, str]:
        # Try new SDK first
        try:
            import google.genai as genai_new
            self._client_new = genai_new.Client(api_key=self.api_key)
            self._use_new    = True
            return True, ''
        except ImportError:
            pass
        except Exception as e:
            return False, f'google-genai error: {e}'
        # Fallback to old SDK
        try:
            import google.generativeai as genai_old
            genai_old.configure(api_key=self.api_key)
            self._client_old = genai_old.GenerativeModel(
                model_name=self.model_id,
                system_instruction=PromptBuilder.SYSTEM_INSTRUCTION,
                generation_config=self._gen_cfg,
            )
            self._use_new = False
            return True, ''
        except ImportError:
            return False, 'No Gemini SDK. Run: pip install google-genai'
        except Exception as e:
            return False, f'google-generativeai error: {e}'

    async def generate(self, user_msg: str) -> str:
        loop = asyncio.get_event_loop()
        if self._use_new:
            from google.genai import types
            def _call():
                r = self._client_new.models.generate_content(
                    model=self.model_id, contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=PromptBuilder.SYSTEM_INSTRUCTION,
                        temperature=self._gen_cfg['temperature'],
                        top_p=self._gen_cfg['top_p'],
                        top_k=self._gen_cfg['top_k'],
                        max_output_tokens=self._gen_cfg['max_output_tokens'],
                        response_mime_type='application/json',
                    ),
                )
                return r.text
            return await loop.run_in_executor(None, _call)
        else:
            return await loop.run_in_executor(
                None, lambda: self._client_old.generate_content(user_msg).text)


class GroqClient(ProviderClient):
    """Native async Groq client with JSON-mode enforcement."""

    def __init__(self, api_key: str, model_id: str) -> None:
        super().__init__(api_key, model_id)
        self._client = None

    @property
    def provider_id(self) -> str:
        return 'groq'

    def setup(self) -> Tuple[bool, str]:
        try:
            import groq as groq_sdk
            self._client = groq_sdk.AsyncGroq(api_key=self.api_key)
            return True, ''
        except ImportError:
            return False, 'Groq SDK not installed. Run: pip install groq'
        except Exception as e:
            return False, f'Groq setup error: {e}'

    async def generate(self, user_msg: str) -> str:
        messages = [
            {'role': 'system', 'content': PromptBuilder.SYSTEM_INSTRUCTION},
            {'role': 'user',   'content': user_msg},
        ]
        response = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.10,
            max_tokens=400,
            response_format={'type': 'json_object'},  # Groq JSON mode
        )
        return response.choices[0].message.content or ''


class OpenRouterClient(ProviderClient):
    """OpenAI-compatible async client pointed at OpenRouter's API."""

    _BASE_URL = 'https://openrouter.ai/api/v1'

    def __init__(self, api_key: str, model_id: str) -> None:
        super().__init__(api_key, model_id)
        self._client = None

    @property
    def provider_id(self) -> str:
        return 'openrouter'

    def setup(self) -> Tuple[bool, str]:
        try:
            import openai as openai_sdk
            self._client = openai_sdk.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self._BASE_URL,
                default_headers={
                    'HTTP-Referer': 'https://github.com/hpat-terminal',
                    'X-Title':      'HPAT v6 Trading Terminal',
                },
            )
            return True, ''
        except ImportError:
            return False, 'OpenAI SDK not installed. Run: pip install openai'
        except Exception as e:
            return False, f'OpenRouter setup error: {e}'

    async def generate(self, user_msg: str) -> str:
        messages = [
            {'role': 'system', 'content': PromptBuilder.SYSTEM_INSTRUCTION},
            {'role': 'user',   'content': user_msg},
        ]
        # Some OpenRouter models support JSON mode, others don't — try with, fallback without
        try:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.10,
                max_tokens=400,
                response_format={'type': 'json_object'},
            )
        except Exception:
            response = await self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.10,
                max_tokens=400,
            )
        return response.choices[0].message.content or ''


def _make_client(provider_id: str, api_key: str, model_id: str) -> ProviderClient:
    """Factory — instantiate the right client for a given provider."""
    if provider_id == 'gemini':
        return GeminiClient(api_key, model_id)
    elif provider_id == 'groq':
        return GroqClient(api_key, model_id)
    elif provider_id == 'openrouter':
        return OpenRouterClient(api_key, model_id)
    else:
        raise ValueError(f'Unknown provider: {provider_id!r}')


# ─── AI ENGINE ────────────────────────────────────────────────────────────────

class AIEngine:
    """
    Provider-agnostic prediction engine.

    Lifecycle:
    1. UI calls configure(provider_id, api_key, model_id) → returns (ok, err)
    2. If ok, UI calls start(loop, feed_stop_event) → schedules run() on feed loop
    3. run() fires _cycle() per cadence (10s volatile, 30s ranging)
    4. _cycle() dedup-checks → calls provider API → emits 'ai_prediction' event
    5. UI calls stop() → exits cleanly
    """

    def __init__(self) -> None:
        self.provider_id: str  = 'gemini'
        self.model_id:    str  = 'gemini-2.0-flash'
        self._client:     Optional[ProviderClient] = None
        self._task:       Optional[asyncio.Task]   = None
        self._stop_ev:    Optional[asyncio.Event]  = None
        self._loop:       Optional[asyncio.AbstractEventLoop] = None
        self._in_flight:  bool  = False
        self._last_result: Optional[PredictionResult] = None
        self._last_feat:  Dict  = {}
        self._call_count: int   = 0
        self._skip_count: int   = 0
        self._error_count: int  = 0
        self._last_latency: float = 0.0
        self._tokens_saved: int = 0
        self.enabled:     bool  = False
        self._last_error: str   = ''

    # ── Configure ─────────────────────────────────────────────────────────────

    def configure(self, provider_id: str, api_key: str,
                  model_id: str) -> Tuple[bool, str]:
        """
        Build and validate the provider client.
        Returns (success, error_message).
        Does NOT start the prediction loop.
        """
        if not api_key or not api_key.strip():
            return False, 'API key is empty'
        if provider_id not in PROVIDERS:
            return False, f'Unknown provider: {provider_id!r}'

        try:
            client = _make_client(provider_id, api_key, model_id)
        except Exception as e:
            return False, str(e)

        ok, err = client.setup()
        if ok:
            self._client     = client
            self.provider_id = provider_id
            self.model_id    = model_id
            self.enabled     = True
            self._last_error = ''
            self._last_feat  = {}   # reset dedup
        else:
            self._client = None
            self.enabled = False
            self._last_error = err
        return ok, err

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop,
              feed_stop_event: asyncio.Event) -> None:
        """Schedule the prediction coroutine onto the feed asyncio loop (thread-safe)."""
        if not self.enabled or not self._client:
            return
        self._loop = loop
        asyncio.run_coroutine_threadsafe(
            self._start_on_loop(feed_stop_event), loop)

    async def _start_on_loop(self, feed_stop_event: asyncio.Event) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        self._stop_ev = asyncio.Event()
        self._task    = asyncio.create_task(self.run(feed_stop_event))

    def stop(self) -> None:
        self.enabled = False
        if self._loop and self._stop_ev:
            self._loop.call_soon_threadsafe(self._stop_ev.set)
        if self._loop and self._task:
            self._loop.call_soon_threadsafe(self._task.cancel)

    # ── Prediction loop ───────────────────────────────────────────────────────

    async def run(self, feed_stop_event: asyncio.Event) -> None:
        await asyncio.sleep(5)   # warm-up
        while True:
            if feed_stop_event.is_set(): break
            if self._stop_ev and self._stop_ev.is_set(): break
            if not self.enabled: break
            if not self._in_flight:
                asyncio.create_task(self._cycle())
            regime  = self._last_feat.get('_regime', 'RANGING')
            cadence = _CADENCE.get(regime, 30)
            try:
                await asyncio.wait_for(
                    asyncio.shield(feed_stop_event.wait()), timeout=cadence)
                break
            except asyncio.TimeoutError:
                pass

    async def _cycle(self) -> None:
        self._in_flight = True
        pair = STATE.pair
        t0   = time.monotonic()
        try:
            feat = FeatureExtractor.extract(pair)
            changed, reason = FeatureExtractor.significant_change(self._last_feat, feat)
            if not changed:
                self._skip_count  += 1
                self._tokens_saved += 200
                if self._last_result:
                    skipped = PredictionResult(
                        **{k: v for k, v in self._last_result.__dict__.items()
                           if k not in ('skipped', 'skip_reason', 'timestamp')},
                        timestamp=feat.get('_ts', ''),
                        skipped=True, skip_reason=reason,
                    )
                    BUS.emit(Event('ai_prediction', skipped))
                return

            user_msg = PromptBuilder.build_user(feat)
            raw      = await self._client.generate(user_msg)
            result   = ResponseParser.parse(raw, feat,
                                             provider=self.provider_id,
                                             model_id=self.model_id)
            self._last_latency = (time.monotonic() - t0) * 1000
            self._last_result  = result
            self._last_feat    = feat
            self._call_count  += 1
            if result.error: self._error_count += 1
            BUS.emit(Event('ai_prediction', result))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._error_count += 1
            self._last_error   = str(e)
            BUS.emit(Event('ai_prediction_error', str(e)))
        finally:
            self._in_flight = False

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        total     = self._call_count + self._skip_count
        skip_rate = self._skip_count / max(total, 1) * 100
        prov_info = PROVIDERS.get(self.provider_id, {})
        model_info = ALL_MODELS.get(self.model_id, {})
        return {
            'calls':          self._call_count,
            'skips':          self._skip_count,
            'skip_rate_pct':  round(skip_rate, 1),
            'errors':         self._error_count,
            'latency_ms':     round(self._last_latency, 0),
            'tokens_saved':   self._tokens_saved,
            'in_flight':      self._in_flight,
            'enabled':        self.enabled,
            'provider':       self.provider_id,
            'provider_name':  prov_info.get('name', ''),
            'model':          self.model_id,
            'model_badge':    model_info.get('badge', ''),
            'model_tier':     model_info.get('tier', ''),
            'last_error':     self._last_error,
        }

    def user_prompt_size(self) -> Tuple[int, int]:
        try:
            feat = FeatureExtractor.extract(STATE.pair)
            msg  = PromptBuilder.build_user(feat)
            return len(msg), len(msg) // 4
        except Exception:
            return 0, 0


# ─── SINGLETON ────────────────────────────────────────────────────────────────
AI = AIEngine()

# Keep gemini_engine.py compatibility alias
GEMINI = AI
