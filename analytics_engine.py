"""
HPAT v6 — analytics_engine.py
All indicator math, order-flow calculations, candle management, and alert logic.
Uses NumPy vectorised operations for RSI/ATR.
All financial math uses Decimal; display-only values remain float for speed.
"""

from __future__ import annotations
import collections
import datetime
import math
import threading
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import numpy as np

from models import (
    STATE, BUS, ACC,
    PAIRS, DECIMALS, BASE_PRICES,
    AlertRecord, CVDPoint, LiqEvent, OHLCVCandle,
    PricePoint, Position, TradeRecord, Event,
)


# ─── UTILITY ──────────────────────────────────────────────────────────────────

def now_ms() -> int:
    import time
    return int(time.time() * 1000)


def fmt(n, d: int = 2) -> str:
    if not n:
        return '--'
    try:
        return f'{n:,.{d}f}'
    except Exception:
        return '--'


def fmt_k(n: float) -> str:
    if n >= 1e9: return f'${n/1e9:.2f}B'
    if n >= 1e6: return f'${n/1e6:.2f}M'
    if n >= 1e3: return f'${n/1e3:.1f}k'
    return f'${n:.0f}'


def fmt_oi(n: float) -> str:
    if n >= 1e9: return f'{n/1e9:.3f}B'
    if n >= 1e6: return f'{n/1e6:.2f}M'
    return f'{n/1e3:.1f}K'


# ─── ORDER BOOK IMBALANCE ─────────────────────────────────────────────────────

def calc_obi(bids, asks) -> float:
    if not bids or not asks:
        return 0.0
    depth = min(20, max(len(bids), len(asks)))
    vb = sum(float(x[1]) for x in bids[:depth])
    va = sum(float(x[1]) for x in asks[:depth])
    return (vb - va) / (vb + va + 1e-9)


# ─── ATR ──────────────────────────────────────────────────────────────────────

def calc_atr(pair: str, tf: str = '5m') -> float:
    candles = list(STATE.candles[pair][tf])
    if len(candles) < 15:
        return 0.0
    candles = candles[-15:]
    atr = 0.0
    for i in range(1, len(candles)):
        c = candles[i]; p = candles[i - 1]
        tr = max(c.h - c.l, abs(c.h - p.c), abs(c.l - p.c))
        atr = tr if i == 1 else (atr * 13 + tr) / 14
    return atr


# ─── RSI (NumPy vectorised) ───────────────────────────────────────────────────

def calc_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    arr   = np.array(prices, dtype=np.float64)
    delta = np.diff(arr)
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


# ─── KELLY CRITERION ─────────────────────────────────────────────────────────

def calc_kelly(balance: Decimal, win_rate: float,
               win_loss_ratio: float) -> Tuple[Decimal, float, float]:
    f  = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    hk = max(0.0, f / 2.0)
    pos = balance * Decimal(str(hk))
    return pos, f, hk


# ─── VWAP ─────────────────────────────────────────────────────────────────────

def get_vwap() -> Tuple[float, float, float, float, float]:
    if STATE.vwap_sum_v == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    vwap     = STATE.vwap_sum_pv / STATE.vwap_sum_v
    variance = (STATE.vwap_sum_pv2 / STATE.vwap_sum_v) - vwap ** 2
    sigma    = math.sqrt(max(variance, 0))
    return vwap, vwap + sigma, vwap + 2 * sigma, vwap - sigma, vwap - 2 * sigma


# ─── POC ─────────────────────────────────────────────────────────────────────

def get_poc() -> Tuple[float, float]:
    if not STATE.vpvr:
        return 0.0, 0.0
    poc_bin = max(STATE.vpvr, key=STATE.vpvr.get)  # type: ignore[arg-type]
    poc     = float(poc_bin)
    dist    = abs(STATE.price - poc) / max(poc, 1) * 100 if STATE.price else 0
    return poc, dist


# ─── MARKET REGIME ────────────────────────────────────────────────────────────

def market_regime(pair: str) -> Tuple[str, str]:
    from models import C
    if len(list(STATE.candles[pair]['5m'])) < 10:
        return 'RANGING', C['blue']
    atr5  = calc_atr(pair, '5m')
    atr30 = calc_atr(pair, '30m') or atr5
    if atr5 > atr30 * 1.5:
        return 'VOLATILE', C['red']
    elif atr5 > atr30 * 1.1:
        return 'TRENDING', C['green']
    else:
        return 'RANGING', C['blue']


# ─── COMPOSITE SIGNAL ─────────────────────────────────────────────────────────

def composite_signal() -> Tuple[str, str]:
    from models import C
    p = STATE.price
    if not p:
        return 'AWAITING', C['text3']
    vwap = STATE.vwap_val
    rsi  = STATE.rsi_1m
    cvd  = STATE.cvd
    obi  = STATE.dm_obi.get(STATE.pair, 0.0)
    score = 0
    if vwap and p > vwap:   score += 1
    elif vwap and p < vwap: score -= 1
    if rsi > 60:  score += 1
    elif rsi < 40: score -= 1
    if cvd > 0:   score += 1
    else:         score -= 1
    if obi > 0.2:   score += 1
    elif obi < -0.2: score -= 1
    if score >= 3:    return '▲ STRONG LONG',  C['green']
    elif score >= 1:  return '↑ LONG BIAS',    C['teal']
    elif score <= -3: return '▼ STRONG SHORT', C['red']
    elif score <= -1: return '↓ SHORT BIAS',   C['amber']
    else:             return '◆ NO CLEAR EDGE', C['text3']


# ─── OI SIGNAL ────────────────────────────────────────────────────────────────

def oi_signal() -> Tuple[str, str]:
    from models import C
    if not STATE.oi_history or len(STATE.oi_history) < 2:
        return 'Awaiting OI data...', C['text3']
    oi  = STATE.oi
    h1  = [x for x in STATE.oi_history if now_ms() - x.t < 3_600_000]
    if len(h1) < 2:
        return 'Collecting data...', C['text3']
    chg   = (oi - h1[0].oi) / max(h1[0].oi, 1) * 100
    hist  = list(STATE.corr_prices[STATE.pair])
    p_up  = STATE.price > hist[0].price if hist else False
    oi_up = chg > 0
    if p_up and oi_up:     return '▲ TREND BUILDING — New money entering', C['green']
    if p_up and not oi_up: return '⚠ SHORT SQUEEZE — Weak, likely reversal', C['amber']
    if not p_up and oi_up: return '▼ TREND BUILDING — New shorts entering', C['red']
    return 'LONG SQUEEZE — Longs being flushed', C['text2']


# ─── CANDLE ENGINE ────────────────────────────────────────────────────────────

def feed_candle(price: float, vol: float, t: int) -> None:
    for tf, ms in [('1m', 60_000), ('5m', 300_000), ('30m', 1_800_000)]:
        bucket = int(t / ms) * ms
        c = STATE.candles[STATE.pair][tf]
        if c and c[-1].t == bucket:
            entry      = c[-1]
            entry.h    = max(entry.h, price)
            entry.l    = min(entry.l, price)
            entry.c    = price
            entry.v   += vol
        else:
            c.append(OHLCVCandle(t=bucket, o=price, h=price,
                                  l=price, c=price, v=vol))
            if len(c) > 1:
                c[-2].closed = True


# ─── ABSORPTION DETECTION ─────────────────────────────────────────────────────

def check_absorption(price: float) -> None:
    if len(STATE.cvd_history) < 30 or STATE.absorption_alerted:
        return
    window    = list(STATE.cvd_history)[-30:]
    cvd_change = abs(window[-1].cvd - window[0].cvd)
    price_chg  = abs(price - window[0].price) / max(window[0].price, 1) * 100
    if cvd_change > STATE.vbuy * 0.05 and price_chg < 0.05:
        STATE.absorption_alerted = True
        is_bearish = window[-1].cvd > window[0].cvd
        msg = ('⚠ BEARISH ABSORB — Limit seller absorbing market buys' if is_bearish
               else '⚠ BULLISH ABSORB — Limit buyer absorbing market sells')
        add_alert('ABSORB', msg, 'red' if is_bearish else 'green', cooldown=20_000)
        threading.Timer(12.0,
            lambda: setattr(STATE, 'absorption_alerted', False)).start()
    elif price_chg > 0.08:
        STATE.absorption_alerted = False


# ─── ALERT ENGINE ─────────────────────────────────────────────────────────────

def add_alert(atype: str, msg: str, color: str = 'amber',
              cooldown: Optional[int] = None) -> None:
    cd  = cooldown or {'POC': 15_000, 'ABSORB': 20_000, 'SPIKE': 5_000,
                       'LIQ': 3_000}.get(atype, 10_000)
    now = now_ms()
    if STATE.alert_cooldown.get(atype, 0) and now - STATE.alert_cooldown[atype] < cd:
        return
    STATE.alert_cooldown[atype] = now
    STATE.alert_count += 1
    ts = datetime.datetime.utcnow().strftime('%H:%M:%S')
    record = AlertRecord(type=atype, msg=msg, color=color, ts=ts)
    STATE.alerts.appendleft(record)
    BUS.emit(Event('alert', record))


# ─── TRADE HANDLER ────────────────────────────────────────────────────────────

def handle_trade(price: float, qty: float, is_buy: bool) -> None:
    """Process a single aggTrade tick — updates all state, emits events."""
    vol = price * qty
    t   = now_ms()

    STATE.prev_price = STATE.price
    STATE.price = price

    if is_buy:
        STATE.cvd  += vol
        STATE.vbuy += vol
    else:
        STATE.cvd   -= vol
        STATE.vsell += vol

    STATE.cvd_history.append(CVDPoint(cvd=STATE.cvd, price=price, t=t))
    STATE.trade_win.append({'t': t, 'vol': vol, 'isBuy': is_buy})

    # Prune trade window
    cutoff = t - 5_000
    while STATE.trade_win and STATE.trade_win[0]['t'] < cutoff:
        STATE.trade_win.popleft()

    STATE.ring_buf.append(price)
    STATE.atr_prices.append(PricePoint(price=price, t=t))
    STATE.atr5m_prices.append(PricePoint(price=price, t=t))

    if price > STATE.high30 or STATE.high30 == 0: STATE.high30 = price
    if price < STATE.low30:   STATE.low30 = price
    if price > STATE.adr_high: STATE.adr_high = price
    if price < STATE.adr_low:  STATE.adr_low  = price

    # VPVR binning
    bin_size = (0.5 if STATE.pair == 'SOLUSDT' else
                5   if STATE.pair == 'ETHUSDT' else
                0.001 if STATE.pair == 'XRPUSDT' else 50)
    b = round(price / bin_size) * bin_size
    STATE.vpvr[b] = STATE.vpvr.get(b, 0.0) + vol

    # Footprint
    if b not in STATE.footprint:
        STATE.footprint[b] = {'buy': 0.0, 'sell': 0.0}
    if is_buy: STATE.footprint[b]['buy']  += vol
    else:      STATE.footprint[b]['sell'] += vol

    # Trim footprint to 20 closest bins
    if len(STATE.footprint) > 30:
        keys_by_dist = sorted(STATE.footprint.keys(),
                               key=lambda x: abs(x - price))
        for k in keys_by_dist[20:]:
            del STATE.footprint[k]

    # Correlation prices
    STATE.corr_prices[STATE.pair].append(PricePoint(price=price, t=t))
    STATE.dm_prices[STATE.pair] = price
    if is_buy: STATE.dm_cvd[STATE.pair] = STATE.dm_cvd.get(STATE.pair, 0.0) + vol
    else:      STATE.dm_cvd[STATE.pair] = STATE.dm_cvd.get(STATE.pair, 0.0) - vol

    # VWAP (typical price = close here since no H/L per tick)
    tp = price
    STATE.vwap_sum_pv  += tp * vol
    STATE.vwap_sum_v   += vol
    STATE.vwap_sum_pv2 += (tp ** 2) * vol
    if STATE.vwap_sum_v:
        STATE.vwap_val = STATE.vwap_sum_pv / STATE.vwap_sum_v

    feed_candle(price, vol, t)

    # Trade classification
    thr = ([10_000, 50_000, 200_000] if STATE.pair == 'BTCUSDT' else
           [5_000,  25_000, 100_000] if STATE.pair == 'ETHUSDT' else
           [1_000,  10_000,  50_000])
    if   vol >= thr[2]: tier = 'WHALE'
    elif vol >= thr[1]: tier = 'INST'
    elif vol >= thr[0]: tier = 'DOLPH'
    else:               tier = 'retail'

    import random
    should_add = tier != 'retail' or random.random() < 0.04
    if should_add:
        entry = TradeRecord(
            price=Decimal(str(price)), qty=Decimal(str(qty)),
            vol=Decimal(str(vol)), is_buy=is_buy, tier=tier,
            ts=datetime.datetime.utcnow().strftime('%H:%M:%S'),
        )
        if not STATE.tape_frozen:
            STATE.trades.appendleft(entry)
        else:
            STATE.tape_buffer.append(entry)

    # Volume spike alert
    if vol > 200_000:
        add_alert('SPIKE', f'Vol spike {fmt_k(vol)} {"BUY" if is_buy else "SELL"}',
                  'green' if is_buy else 'red')

    check_absorption(price)

    # Emit event so decoupled UI panels can react
    BUS.emit(Event('trade', {'price': price, 'qty': qty, 'is_buy': is_buy,
                              'vol': vol, 'tier': tier}))


# ─── RSI REFRESH (called periodically from analytics thread) ──────────────────

def refresh_rsi() -> None:
    prices = [x.price for x in list(STATE.atr5m_prices)[-30:]]
    if len(prices) > 15:
        STATE.rsi_1m = calc_rsi(prices)
        STATE.rsi_5m = calc_rsi(prices[::2] if len(prices) > 20 else prices)


# ─── P&L CALCULATION (Decimal FIFO) ──────────────────────────────────────────

def commission_to_usdt(commission: Decimal, comm_asset: str,
                        trade_price: Decimal, trade_symbol: str) -> Decimal:
    """Convert commission to USDT using Decimal arithmetic throughout."""
    if comm_asset == 'USDT':
        return commission
    base_asset = trade_symbol.replace('USDT', '').replace('BUSD', '')
    if comm_asset == base_asset:
        return commission * trade_price
    if comm_asset == 'BNB':
        bnb_price = Decimal(str(STATE.dm_prices.get('BNBUSDT',
                                float(BASE_PRICES.get('BNBUSDT', Decimal('580'))))))
        return commission * bnb_price
    return commission * trade_price


def calc_pnl(trades) -> None:
    """
    FIFO P&L using Decimal for every monetary value.
    Writes results directly to ACC.
    """
    buy_queues: Dict[str, collections.deque] = {}
    pnl_map:   Dict[str, Decimal] = {}
    win = 0; loss = 0
    best = None; worst = None

    for t in trades:
        sym = t.symbol
        if sym not in buy_queues:
            buy_queues[sym] = collections.deque()
            pnl_map[sym]    = Decimal('0')

        comm_usdt = commission_to_usdt(t.commission, t.commAsset, t.price, sym)

        if t.side == 'BUY':
            qty = t.qty
            if qty < Decimal('1e-12'):
                continue
            total_cost    = t.quoteQty + comm_usdt
            cost_per_unit = total_cost / qty
            buy_queues[sym].append([cost_per_unit, qty])

        else:
            qty_to_sell    = t.qty
            total_revenue  = t.quoteQty - comm_usdt
            rev_per_unit   = total_revenue / max(qty_to_sell, Decimal('1e-12'))
            trade_pnl      = Decimal('0')

            while qty_to_sell > Decimal('1e-9') and buy_queues[sym]:
                cost_per_unit, buy_qty = buy_queues[sym][0]
                matched     = min(qty_to_sell, buy_qty)
                trade_pnl  += matched * (rev_per_unit - cost_per_unit)
                qty_to_sell -= matched
                buy_queues[sym][0][1] -= matched
                if buy_queues[sym][0][1] < Decimal('1e-9'):
                    buy_queues[sym].popleft()

            pnl_map[sym] = pnl_map.get(sym, Decimal('0')) + trade_pnl
            t.pnl = trade_pnl

            if trade_pnl >= 0:
                win += 1
            else:
                loss += 1
            if best is None or trade_pnl > best.pnl:
                best = t
            if worst is None or trade_pnl < worst.pnl:
                worst = t

    ACC.realized_pnl = pnl_map
    ACC.total_pnl    = sum(pnl_map.values(), Decimal('0'))
    ACC.win_trades   = win
    ACC.loss_trades  = loss
    ACC.best_trade   = best
    ACC.worst_trade  = worst
