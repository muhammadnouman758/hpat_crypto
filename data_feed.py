"""
HPAT v6 — data_feed.py
Async WebSocket + REST data feed using asyncio / aiohttp / websockets.
Implements exponential backoff with jitter for reconnection.
Runs in a dedicated background thread that owns its own event loop.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from decimal import Decimal
from typing import Optional

try:
    import aiohttp
    import websockets
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

from models import STATE, BUS, ACC, PAIRS, BASE_PRICES, Event
from analytics_engine import (
    handle_trade, calc_obi, add_alert, now_ms,
    commission_to_usdt, calc_pnl,
)

# ─── BINANCE ENDPOINTS ────────────────────────────────────────────────────────
BINANCE_REST  = 'https://api.binance.com'
BINANCE_FAPI  = 'https://fapi.binance.com'
BINANCE_WS    = 'wss://stream.binance.com:9443/stream'

# ─── BACKOFF CONFIG ────────────────────────────────────────────────────────────
_BASE_DELAY   = 1.0   # seconds
_MAX_DELAY    = 120.0 # seconds
_JITTER_SCALE = 0.3   # ±30 % jitter


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with full jitter: random(0, min(cap, base * 2^n))."""
    cap   = min(_MAX_DELAY, _BASE_DELAY * (2 ** attempt))
    delay = random.uniform(0, cap)
    return delay


# ─── REST API HELPERS ─────────────────────────────────────────────────────────

import hmac
import hashlib
import urllib.parse


def _sign(params: dict, secret: str) -> str:
    query = urllib.parse.urlencode(params)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


async def _binance_get(session: 'aiohttp.ClientSession', endpoint: str,
                        params: dict = None, signed: bool = False,
                        futures: bool = False) -> dict:
    base = BINANCE_FAPI if futures else BINANCE_REST
    p = dict(params or {})
    if signed:
        p['timestamp'] = int(time.time() * 1000)
        p['signature'] = _sign(p, ACC.api_secret)
    headers = {}
    if ACC.api_key:
        headers['X-MBX-APIKEY'] = ACC.api_key
    async with session.get(f'{base}{endpoint}', params=p,
                            headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as r:
        return await r.json()


# ─── OI / FUNDING FETCH ───────────────────────────────────────────────────────

async def fetch_oi(session: 'aiohttp.ClientSession') -> None:
    try:
        d = await _binance_get(session,
                                f'/fapi/v1/openInterest',
                                params={'symbol': STATE.pair}, futures=True)
        STATE.oi = float(d['openInterest']) * STATE.price
        from models import OIPoint
        STATE.oi_history.append(OIPoint(oi=STATE.oi, t=now_ms()))
    except Exception:
        pass


async def fetch_funding(session: 'aiohttp.ClientSession') -> None:
    try:
        d = await _binance_get(session,
                                '/fapi/v1/premiumIndex',
                                params={'symbol': STATE.pair}, futures=True)
        STATE.funding    = float(d['lastFundingRate']) * 100
        STATE.funding_cd = int(d['nextFundingTime']) - now_ms()
        if not STATE.funding_history or STATE.funding_history[-1] != STATE.funding:
            STATE.funding_history.append(STATE.funding)
    except Exception:
        pass


async def _periodic_rest(session: 'aiohttp.ClientSession') -> None:
    """Background coroutine — polls OI and funding every 30 s."""
    while True:
        await fetch_oi(session)
        await fetch_funding(session)
        await asyncio.sleep(30)


# ─── LIVE WEBSOCKET FEED ──────────────────────────────────────────────────────

def _build_ws_url(pair: str) -> str:
    p = pair.lower()
    return (f'{BINANCE_WS}?streams='
            f'{p}@aggTrade/{p}@bookTicker/{p}@depth20@100ms')


async def _process_ws_message(msg: str) -> None:
    d      = json.loads(msg)
    stream = d.get('stream', '')
    data   = d.get('data', d)

    if 'aggTrade' in stream:
        price  = float(data['p'])
        qty    = float(data['q'])
        is_buy = not data['m']
        handle_trade(price, qty, is_buy)

    elif 'bookTicker' in stream:
        STATE.bid = float(data['b'])
        STATE.ask = float(data['a'])

    elif 'depth' in stream:
        if 'bids' in data: STATE.ob_bids = data['bids']
        if 'asks' in data: STATE.ob_asks = data['asks']
        STATE.dm_obi[STATE.pair] = calc_obi(STATE.ob_bids, STATE.ob_asks)
        BUS.emit(Event('orderbook', None))


async def _ws_connect_loop(stop_event: asyncio.Event) -> None:
    attempt = 0
    while not stop_event.is_set():
        url = _build_ws_url(STATE.pair)
        try:
            STATE.ws_status = 'CONNECTING'
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                attempt = 0          # reset backoff on successful connect
                STATE.ws_status = 'LIVE'
                BUS.emit(Event('ws_status', 'LIVE'))
                async for msg in ws:
                    if stop_event.is_set():
                        break
                    await _process_ws_message(msg)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            if stop_event.is_set():
                break
            STATE.ws_status = 'RECONNECTING'
            BUS.emit(Event('ws_status', 'RECONNECTING'))
            delay = _backoff_delay(attempt)
            attempt += 1
            await asyncio.sleep(delay)


# ─── ACCOUNT REST CALLS ───────────────────────────────────────────────────────

async def acc_fetch_balances(session: 'aiohttp.ClientSession') -> None:
    from models import Balance
    data = await _binance_get(session, '/api/v3/account', signed=True)
    ticker_data = await _binance_get(session, '/api/v3/ticker/price')
    prices = {t['symbol']: Decimal(t['price']) for t in ticker_data}

    balances = []
    total    = Decimal('0')
    for b in data.get('balances', []):
        free   = Decimal(b['free'])
        locked = Decimal(b['locked'])
        total_qty = free + locked
        if total_qty < Decimal('1e-9'):
            continue
        asset = b['asset']
        usd   = Decimal('0')
        if asset == 'USDT':
            usd = total_qty
        elif asset + 'USDT' in prices:
            usd = total_qty * prices[asset + 'USDT']
        elif asset == 'BTC':
            usd = total_qty * prices.get('BTCUSDT', Decimal('0'))
        if usd < Decimal('0.01'):
            continue
        balances.append(Balance(asset=asset, free=free, locked=locked,
                                 total=total_qty, usd_val=usd))
        total += usd

    balances.sort(key=lambda x: x.usd_val, reverse=True)
    ACC.balances  = balances
    ACC.total_usd = total


async def acc_fetch_open_orders(session: 'aiohttp.ClientSession') -> None:
    import datetime as dt
    from models import OpenOrder
    data = await _binance_get(session, '/api/v3/openOrders', signed=True)
    orders = []
    for o in data:
        orders.append(OpenOrder(
            symbol=o['symbol'], side=o['side'], type=o['type'],
            price=Decimal(o['price']), qty=Decimal(o['origQty']),
            filled=Decimal(o['executedQty']), status=o['status'],
            time=dt.datetime.utcfromtimestamp(o['time'] / 1000).strftime('%m-%d %H:%M'),
            order_id=int(o['orderId']),
        ))
    ACC.open_orders = orders


async def acc_fetch_trade_history(session: 'aiohttp.ClientSession') -> None:
    import datetime as dt
    from models import TradeHistoryEntry
    all_trades = []
    for sym in PAIRS:
        try:
            data = await _binance_get(session, '/api/v3/myTrades',
                                       params={'symbol': sym, 'limit': 100}, signed=True)
            for t in data:
                all_trades.append(TradeHistoryEntry(
                    symbol=t['symbol'],
                    side='BUY' if t['isBuyer'] else 'SELL',
                    price=Decimal(t['price']),
                    qty=Decimal(t['qty']),
                    quoteQty=Decimal(t['quoteQty']),
                    commission=Decimal(t['commission']),
                    commAsset=t['commissionAsset'],
                    time=dt.datetime.utcfromtimestamp(t['time'] / 1000).strftime('%m-%d %H:%M'),
                    time_ms=int(t['time']),
                ))
        except Exception:
            pass
    all_trades.sort(key=lambda x: x.time_ms)
    ACC.trade_history = all_trades
    calc_pnl(all_trades)


async def acc_cancel_order_async(session: 'aiohttp.ClientSession',
                                  symbol: str, order_id: int):
    import time as _time
    p = {'symbol': symbol, 'orderId': order_id,
         'timestamp': int(_time.time() * 1000)}
    p['signature'] = _sign(p, ACC.api_secret)
    headers = {'X-MBX-APIKEY': ACC.api_key}
    async with session.delete(f'{BINANCE_REST}/api/v3/order',
                               params=p, headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
        return await r.json()


async def acc_refresh_all(session: 'aiohttp.ClientSession') -> None:
    if ACC.refreshing:
        return
    ACC.refreshing = True
    ACC.error = ''
    try:
        await acc_fetch_balances(session)
        await acc_fetch_open_orders(session)
        await acc_fetch_trade_history(session)
        ACC.connected    = True
        ACC.last_refresh = time.time()
    except Exception as e:
        ACC.connected = False
        ACC.error     = str(e)
    finally:
        ACC.refreshing = False
    BUS.emit(Event('acc_refresh', None))


# ─── SIMULATION ENGINE ────────────────────────────────────────────────────────

async def run_sim_async(stop_event: asyncio.Event) -> None:
    """Async simulation — replaces the blocking threaded sim of v5."""
    import random as _r
    from models import OIPoint
    from analytics_engine import calc_rsi

    STATE.ws_status = 'SIM'
    BUS.emit(Event('ws_status', 'SIM'))

    for p in PAIRS:
        if STATE.dm_prices[p] == 0.0:
            STATE.dm_prices[p] = float(BASE_PRICES[p])

    tick = 0
    while not stop_event.is_set():
        tick += 1
        base = float(STATE.base_prices[STATE.pair])
        base += (_r.random() - 0.499) * base * 0.00025
        STATE.base_prices[STATE.pair] = Decimal(str(base))
        STATE.bid = base - base * 0.00008 * (_r.random() + 0.5)
        STATE.ask = base + base * 0.00008 * (_r.random() + 0.5)
        qty    = _r.random() * 2 + 0.01
        is_buy = _r.random() > 0.49
        handle_trade(base, qty, is_buy)

        if tick % 5 == 0:
            STATE.ob_bids = [[(base - base * 0.0001 * (i + 1)),
                               round(_r.random() * 8 + 0.1, 4)]
                              for i in range(10)]
            STATE.ob_asks = [[(base + base * 0.0001 * (i + 1)),
                               round(_r.random() * 8 + 0.1, 4)]
                              for i in range(10)]
            STATE.dm_obi[STATE.pair] = calc_obi(STATE.ob_bids, STATE.ob_asks)

        if tick % 3 == 0:
            for pp in PAIRS:
                if pp == STATE.pair: continue
                pb = float(STATE.base_prices.get(pp, BASE_PRICES[pp]))
                pb += (_r.random() - 0.499) * pb * 0.0003
                STATE.base_prices[pp] = Decimal(str(pb))
                STATE.dm_prices[pp]   = pb
                from models import PricePoint
                STATE.corr_prices[pp].append(PricePoint(price=pb, t=now_ms()))

        if tick % 30 == 0:
            base_oi = STATE.price * (800_000 if STATE.pair == 'BTCUSDT' else
                                      2_000_000 if STATE.pair == 'ETHUSDT' else 5_000_000)
            STATE.oi = base_oi + (_r.random() - 0.5) * base_oi * 0.002
            STATE.oi_history.append(OIPoint(oi=STATE.oi, t=now_ms()))
            STATE.funding = (_r.random() - 0.48) * 0.1
            STATE.funding_history.append(STATE.funding)

        # RSI
        prices = [x.price for x in list(STATE.atr5m_prices)[-30:]]
        if len(prices) > 15:
            STATE.rsi_1m = calc_rsi(prices)
            STATE.rsi_5m = calc_rsi(prices[::2] if len(prices) > 20 else prices)

        await asyncio.sleep(0.12)


# ─── FEED CONTROLLER ─────────────────────────────────────────────────────────

class FeedController:
    """
    Manages the asyncio event loop in a background thread.
    Provides start/stop/switch_pair interface to the UI thread.
    """

    def __init__(self) -> None:
        self._loop:       Optional[asyncio.AbstractEventLoop] = None
        self._thread:     Optional[threading.Thread]          = None
        self._stop_event: Optional[asyncio.Event]             = None
        self._sim_mode    = not ASYNC_AVAILABLE

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name='FeedThread')
        self._thread.start()

    def stop(self) -> None:
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        STATE.sim_running = False

    def switch_pair(self, pair: str) -> None:
        """Restart the feed for a new pair."""
        STATE.reset_pair(pair)
        self.stop()
        self.start()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            self._loop.close()

    async def _main(self) -> None:
        self._stop_event = asyncio.Event()

        # Import here to avoid circular import at module level
        from gemini_engine import GEMINI

        if self._sim_mode:
            STATE.sim_running = True
            tasks = [asyncio.create_task(run_sim_async(self._stop_event))]
            if GEMINI.enabled:
                tasks.append(asyncio.create_task(GEMINI.run(self._stop_event)))
            await self._stop_event.wait()
            for t in tasks:
                t.cancel()
            return

        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    asyncio.create_task(_ws_connect_loop(self._stop_event)),
                    asyncio.create_task(_periodic_rest(session)),
                ]
                if GEMINI.enabled:
                    tasks.append(asyncio.create_task(GEMINI.run(self._stop_event)))
                await self._stop_event.wait()
                for t in tasks:
                    t.cancel()
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
        except Exception:
            # Graceful fallback to sim
            STATE.sim_running = True
            sim_task = asyncio.create_task(run_sim_async(self._stop_event))
            if GEMINI.enabled:
                ai_task = asyncio.create_task(GEMINI.run(self._stop_event))
            await self._stop_event.wait()

    def start_gemini(self) -> None:
        """Re-launch the Gemini prediction loop on the existing event loop."""
        from gemini_engine import GEMINI
        if self._loop and GEMINI.enabled:
            asyncio.run_coroutine_threadsafe(
                self._gemini_task_wrapper(), self._loop)

    async def _gemini_task_wrapper(self) -> None:
        from gemini_engine import GEMINI
        if self._stop_event:
            await GEMINI.run(self._stop_event)

    def run_account_refresh(self) -> None:
        """Schedule an account refresh on the feed loop (thread-safe)."""
        if self._loop and ASYNC_AVAILABLE and ACC.api_key:
            asyncio.run_coroutine_threadsafe(
                self._acc_refresh_task(), self._loop)

    async def _acc_refresh_task(self) -> None:
        async with aiohttp.ClientSession() as session:
            await acc_refresh_all(session)

    def cancel_order(self, symbol: str, order_id: int) -> None:
        if self._loop and ASYNC_AVAILABLE:
            asyncio.run_coroutine_threadsafe(
                self._cancel_order_task(symbol, order_id), self._loop)

    async def _cancel_order_task(self, symbol: str, order_id: int) -> None:
        async with aiohttp.ClientSession() as session:
            await acc_cancel_order_async(session, symbol, order_id)
