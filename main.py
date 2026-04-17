"""
HPAT v6 — main.py
Enterprise-grade refactor of the HPAT trading terminal.

Architecture:
  models.py          → Typed state, EventBus, AccountState
  analytics_engine.py → All math (vectorised NumPy RSI/ATR, Decimal P&L)
  data_feed.py        → asyncio WebSocket + aiohttp REST, exponential backoff
  views/             → Decoupled UI view components (object-pool canvas)
  main.py            → MVVM wiring; UI update loop driven by UI_QUEUE / after()

Key improvements over v5:
  ✓ MVC/MVVM decomposition — no God-Object class
  ✓ Pub/Sub EventBus — UI panels subscribe to events, no global polling
  ✓ asyncio + aiohttp + websockets — no blocking urllib / threading.sleep
  ✓ Exponential backoff with jitter for WebSocket reconnection
  ✓ Object-pooled canvas rendering — no delete('all') spam
  ✓ NumPy-vectorised RSI / ATR
  ✓ Decimal arithmetic for all P&L, sizing, cost-basis
  ✓ API keys via env vars / .env — never in UI widgets
"""

from __future__ import annotations

import collections
import datetime
import math
import os
import sys
import threading
import tkinter as tk
from decimal import Decimal
from tkinter import ttk
from typing import Dict, Optional

# ── Core modules ───────────────────────────────────────────────────────────────
from models import (
    C, STATE, ACC, BUS,
    PAIRS, PAIR_LABELS, DECIMALS, BASE_PRICES,
)
from analytics_engine import (
    fmt, fmt_k, fmt_oi,
    calc_obi, calc_atr, calc_rsi, calc_kelly,
    get_vwap, get_poc,
    market_regime, composite_signal, oi_signal,
    add_alert, now_ms,
)
from data_feed import FeedController

# ── View components ────────────────────────────────────────────────────────────
from views.orderbook_view import OrderBookView
from views.vpvr_view      import VPVRView
from views.account_view   import AccountView
from views.ai_prediction_view import AIPredictionView


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ─── MAIN APPLICATION ─────────────────────────────────────────────────────────

class HPAT_App(tk.Tk):
    """
    Root window and MVVM coordinator.
    - Owns the FeedController (data layer)
    - Builds all tab views
    - Runs a 250 ms UI update loop that reads STATE snapshots
    - Subscribes to EventBus for instant UI reactions
    """

    def __init__(self) -> None:
        super().__init__()
        self.title('HPAT v6 — High-Precision Analytical Terminal')
        self.configure(bg=C['bg'])
        self.geometry('1680x960')
        self.minsize(1400, 800)

        self._build_fonts()
        self._build_ui()

        # Animation / smooth state
        self._price_flash_steps = 0
        self._price_flash_color: Optional[str] = None
        self._price_base_color  = C['green']
        self._rsi1m_smooth      = 50.0
        self._rsi5m_smooth      = 50.0
        self._cvd_pct_smooth    = 0.5
        self._dm_flash: Dict    = {}
        self._tape_hash         = None
        self._alert_hash        = None

        # Subscribe to events for instant panel refreshes
        BUS.subscribe('ws_status',  lambda e: self.after(0, self._on_ws_status, e))
        BUS.subscribe('alert',      lambda e: self.after(0, self._on_alert, e))

        # Start data feed
        self._feed = FeedController()
        self._feed.start()

        # Expose feed on app for AccountView
        self._app_feed = self._feed

        # Start main update loop
        self.after(250, self._update_loop)

    # ── Fonts ─────────────────────────────────────────────────────────────────

    def _build_fonts(self) -> None:
        mono = 'Consolas' if sys.platform == 'win32' else 'Courier'
        self.MONO       = mono
        self.F_MONO_SM  = (mono, 10)
        self.F_MONO_MED = (mono, 11)
        self.F_MONO_LG  = (mono, 12, 'bold')
        self.F_BIG      = (mono, 26, 'bold')
        self.F_MED      = (mono, 16, 'bold')
        self.F_LABEL    = (mono, 9)
        self.F_HEAD     = (mono, 10, 'bold')
        self.F_TINY     = (mono, 9)

    # ── UI builders ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_topbar()
        self.nb = ttk.Notebook(self)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=C['bg1'], borderwidth=0)
        style.configure('TNotebook.Tab', background=C['bg2'], foreground=C['text2'],
                        padding=[14, 6], font=self.F_HEAD, borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', C['bg3'])],
                  foreground=[('selected', C['cyan'])])
        self.nb.pack(fill='both', expand=True, padx=2, pady=2)
        self._build_tab_main()
        self._build_tab_orderflow()
        self._build_tab_analytics()
        self._build_tab_risk()
        self._build_tab_dominance()
        self._build_tab_alerts()
        self._build_tab_account()
        self._build_tab_ai()

    def _panel(self, parent, title: str, right_text: str = '', accent=None):
        f   = tk.Frame(parent, bg=C['bg1'], bd=0,
                        highlightthickness=1, highlightbackground=C['border2'])
        hdr = tk.Frame(f, bg=C['bg2'])
        hdr.pack(fill='x')
        acc_color = accent or C['cyan']
        tk.Frame(hdr, bg=acc_color, width=3).pack(side='left', fill='y')
        tk.Label(hdr, text='  ' + title, bg=C['bg2'], fg=C['text'],
                 font=self.F_HEAD, anchor='w', padx=4, pady=6).pack(side='left')
        if right_text:
            tk.Label(hdr, text=right_text + '  ', bg=C['bg2'], fg=acc_color,
                     font=self.F_TINY, anchor='e', padx=4).pack(side='right')
        tk.Frame(f, bg=C['border'], height=1).pack(fill='x')
        body = tk.Frame(f, bg=C['bg1'], padx=10, pady=8)
        body.pack(fill='both', expand=True)
        return f, body

    def _stat_row(self, parent, label: str, var_name: str, color=None):
        r = tk.Frame(parent, bg=C['bg1'])
        r.pack(fill='x', pady=1)
        tk.Label(r, text=label, bg=C['bg1'], fg=C['text3'],
                 font=self.F_LABEL, width=16, anchor='w').pack(side='left')
        v = tk.StringVar(value='--')
        l = tk.Label(r, textvariable=v, bg=C['bg1'], fg=color or C['text'],
                     font=self.F_MONO_SM, anchor='e')
        l.pack(side='right')
        setattr(self, var_name + '_var', v)
        setattr(self, var_name + '_lbl', l)
        return v, l

    def _build_topbar(self) -> None:
        tb = tk.Frame(self, bg=C['bg1'], pady=6, padx=8)
        tb.pack(fill='x')
        tk.Label(tb, text='  HPAT', bg=C['bg1'], fg=C['cyan'],
                 font=(self.MONO, 16, 'bold')).pack(side='left', padx=(0, 2))
        tk.Label(tb, text='// v6 PRO', bg=C['bg1'], fg=C['text3'],
                 font=(self.MONO, 11)).pack(side='left', padx=(0, 16))
        self._pair_btns: Dict[str, tk.Button] = {}
        for p in PAIRS:
            b = tk.Button(tb, text=PAIR_LABELS[p], bg=C['bg2'], fg=C['text2'],
                          font=(self.MONO, 11, 'bold'), bd=0, relief='flat',
                          padx=12, pady=5, activebackground=C['cyan'],
                          activeforeground='#000', cursor='hand2',
                          command=lambda pp=p: self._switch_pair(pp))
            b.pack(side='left', padx=2)
            self._pair_btns[p] = b
        self._pair_btns['BTCUSDT'].config(bg=C['cyan'], fg='#000')

        self._ws_status_var = tk.StringVar(value='INIT')
        self._funding_var   = tk.StringVar(value='--')
        self._clock_var     = tk.StringVar(value='--:--:--')
        for lbl, var in [('FEED', self._ws_status_var),
                          ('FUNDING', self._funding_var),
                          ('UTC', self._clock_var)]:
            f = tk.Frame(tb, bg=C['bg2'], padx=8, pady=4,
                         highlightthickness=1, highlightbackground=C['border'])
            f.pack(side='left', padx=4)
            tk.Label(f, text=lbl, bg=C['bg2'], fg=C['text3'],
                     font=self.F_TINY).pack(side='left', padx=(0, 5))
            tk.Label(f, textvariable=var, bg=C['bg2'], fg=C['text2'],
                     font=self.F_MONO_MED).pack(side='left')

    # ── TAB: DASHBOARD ────────────────────────────────────────────────────────

    def _build_tab_main(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  DASHBOARD  ')
        col_l = tk.Frame(tab, bg=C['bg'], width=310)
        col_l.pack(side='left', fill='y', padx=(2, 1), pady=2)
        col_l.pack_propagate(False)
        col_m = tk.Frame(tab, bg=C['bg'])
        col_m.pack(side='left', fill='both', expand=True, padx=1, pady=2)
        col_r = tk.Frame(tab, bg=C['bg'], width=310)
        col_r.pack(side='left', fill='y', padx=(1, 2), pady=2)
        col_r.pack_propagate(False)

        self._build_price_panel(col_l)
        self._build_vwap_panel(col_l)
        self._build_rsi_panel(col_l)
        self._build_oi_panel(col_l)
        self._build_tape_panel(col_m)
        # Object-pooled order book
        self._ob_view = OrderBookView(col_m, self)
        self._ob_view._pf.pack(fill='both', expand=True, pady=(1, 0))
        # Object-pooled VPVR
        self._vpvr_view = VPVRView(col_r, self)
        self._build_funding_panel(col_r)
        self._build_context_panel(col_r)

    def _build_price_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'PRICE', accent=C['green'])
        pf.pack(fill='x', pady=(0, 1))
        self.price_var     = tk.StringVar(value='--')
        self.price_chg_var = tk.StringVar(value='--')
        self.bid_var       = tk.StringVar(value='--')
        self.ask_var       = tk.StringVar(value='--')
        self.spread_var    = tk.StringVar(value='--')
        self.price_lbl = tk.Label(body, textvariable=self.price_var,
                                   bg=C['bg1'], fg=C['green'],
                                   font=self.F_BIG)
        self.price_lbl.pack(anchor='w')
        tk.Label(body, textvariable=self.price_chg_var,
                 bg=C['bg1'], fg=C['text2'], font=self.F_MONO_MED).pack(anchor='w')
        for lbl, vname, col in [('BID', 'bid_var', C['green']),
                                  ('ASK', 'ask_var', C['red']),
                                  ('SPREAD', 'spread_var', C['text2'])]:
            self._stat_row(body, lbl, vname.replace('_var', ''), col)

    def _build_vwap_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'VWAP + σ BANDS')
        pf.pack(fill='x', pady=(0, 1))
        self.vwap_var     = tk.StringVar(value='--')
        self.vwap_sig_var = tk.StringVar(value='Awaiting data...')
        tk.Label(body, textvariable=self.vwap_var, bg=C['bg1'],
                 fg=C['blue'], font=(self.MONO, 18, 'bold')).pack(anchor='w')
        tk.Label(body, textvariable=self.vwap_sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM).pack(anchor='w', pady=2)
        self.vwap_band_vars: Dict[str, tk.StringVar] = {}
        for k, col in [('+2σ', C['red']), ('+1σ', C['amber']),
                        ('DIST', C['cyan']),
                        ('-1σ', C['teal']), ('-2σ', C['green'])]:
            r = tk.Frame(body, bg=C['bg1']); r.pack(fill='x', pady=1)
            tk.Label(r, text=k, bg=C['bg1'], fg=C['text3'],
                     font=self.F_LABEL, width=6, anchor='w').pack(side='left')
            v = tk.StringVar(value='--')
            tk.Label(r, textvariable=v, bg=C['bg1'], fg=col,
                     font=self.F_MONO_SM, anchor='e').pack(side='right')
            self.vwap_band_vars[k] = v

    def _build_rsi_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'RSI OSCILLATOR')
        pf.pack(fill='x', pady=(0, 1))
        self.rsi_sig_var = tk.StringVar(value='Awaiting data...')
        tk.Label(body, textvariable=self.rsi_sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM, wraplength=280).pack(anchor='w', pady=2)
        for name, col in [('rsi1m', C['cyan']), ('rsi5m', C['blue'])]:
            v, l = self._stat_row(body, name.upper().replace('RSI', 'RSI '), name, col)
            frame = tk.Frame(body, bg=C['bg3'], height=8); frame.pack(fill='x', pady=1)
            bar   = tk.Frame(frame, bg=col, height=8)
            setattr(self, f'{name}_frame', frame)
            setattr(self, f'{name}_bar',   bar)

    def _build_oi_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'OPEN INTEREST', 'FUTURES')
        pf.pack(fill='x', pady=(0, 1))
        self.oi_var      = tk.StringVar(value='--')
        self.oi_chg1h_var = tk.StringVar(value='--')
        self.oi_chg4h_var = tk.StringVar(value='--')
        self.oi_sig_var  = tk.StringVar(value='Awaiting OI data...')
        tk.Label(body, textvariable=self.oi_var, bg=C['bg1'],
                 fg=C['purple'], font=(self.MONO, 18, 'bold')).pack(anchor='w')
        row = tk.Frame(body, bg=C['bg1']); row.pack(fill='x')
        for lbl, vname in [('1H CHG', 'oi_chg1h_var'), ('4H CHG', 'oi_chg4h_var')]:
            f = tk.Frame(row, bg=C['bg2'], padx=8, pady=4,
                         highlightthickness=1, highlightbackground=C['border2'])
            f.pack(side='left', padx=(0, 4), pady=3)
            tk.Label(f, text=lbl, bg=C['bg2'], fg=C['text3'], font=self.F_HEAD).pack()
            v = tk.StringVar(value='--')
            tk.Label(f, textvariable=v, bg=C['bg2'], fg=C['text'],
                     font=(self.MONO, 11, 'bold')).pack()
            setattr(self, vname, v)
        tk.Label(body, textvariable=self.oi_sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM, wraplength=280).pack(anchor='w', pady=3)

    def _build_tape_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'LIVE TAPE', 'REAL-TIME')
        pf.pack(fill='both', expand=True, pady=(0, 1))
        ctrl = tk.Frame(body, bg=C['bg1']); ctrl.pack(fill='x', pady=(0, 4))
        self.tape_status_var = tk.StringVar(value='LIVE')
        self.tps_var = tk.StringVar(value='0.0 t/s')
        tk.Label(ctrl, textvariable=self.tape_status_var, bg=C['bg1'],
                 fg=C['green'], font=(self.MONO, 11, 'bold')).pack(side='left')
        tk.Label(ctrl, textvariable=self.tps_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_MED).pack(side='left', padx=10)
        tk.Button(ctrl, text='❄ FREEZE', bg=C['bg3'], fg=C['cyan'],
                  font=self.F_MONO_SM, bd=0, padx=8, pady=3, cursor='hand2',
                  highlightthickness=1, highlightbackground=C['border'],
                  command=self._toggle_freeze).pack(side='right')
        frame = tk.Frame(body, bg=C['bg1']); frame.pack(fill='both', expand=True)
        sb = tk.Scrollbar(frame, bg=C['bg2'], troughcolor=C['bg3']); sb.pack(side='right', fill='y')
        self.tape_list = tk.Text(frame, bg=C['bg1'], fg=C['text'],
                                  font=self.F_MONO_MED, bd=0,
                                  yscrollcommand=sb.set, state='disabled', height=14,
                                  selectbackground=C['bg3'])
        self.tape_list.pack(fill='both', expand=True)
        sb.config(command=self.tape_list.yview)
        for tag, col in [('buy', C['green']), ('sell', C['red']),
                          ('whale', C['amber']), ('inst', C['purple']),
                          ('dolph', C['blue']), ('ts', C['text3']),
                          ('price', C['text']), ('vol', C['text2'])]:
            self.tape_list.tag_configure(tag, foreground=col)

    def _build_funding_panel(self, parent) -> None:
        pf, body = self._panel(parent, 'FUNDING RATE', 'PERP')
        pf.pack(fill='x', pady=(0, 1))
        self.fund_rate_var = tk.StringVar(value='--')
        self.fund_bias_var = tk.StringVar(value='--')
        self.fund_cd_var   = tk.StringVar(value='--:--:--')
        self.fund_rate_lbl = tk.Label(body, textvariable=self.fund_rate_var,
                                       bg=C['bg1'], fg=C['text'],
                                       font=(self.MONO, 20, 'bold'))
        self.fund_rate_lbl.pack(anchor='w')
        row = tk.Frame(body, bg=C['bg1']); row.pack(fill='x')
        tk.Label(row, textvariable=self.fund_bias_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM).pack(side='left')
        tk.Label(row, text=' | NEXT: ', bg=C['bg1'],
                 fg=C['text3'], font=self.F_LABEL).pack(side='left')
        tk.Label(row, textvariable=self.fund_cd_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM).pack(side='left')

    def _build_context_panel(self, parent) -> None:
        pf, body = self._panel(parent, '⬡ CONTEXT ENGINE v6')
        pf.pack(fill='x', pady=(0, 1))
        self.ctx_regime_var    = tk.StringVar(value='RANGING')
        self.ctx_composite_var = tk.StringVar(value='Awaiting data...')
        self.ctx_adr_var       = tk.StringVar(value='ADR: --')
        self.ctx_vol_var       = tk.StringVar(value='Vol: --x')
        for lbl, vname, col in [
            ('MARKET REGIME',    'ctx_regime_var',    C['blue']),
            ('COMPOSITE SIGNAL', 'ctx_composite_var', C['cyan']),
            ('ADR COMPLETION',   'ctx_adr_var',       C['amber']),
            ('VOL EXPANSION',    'ctx_vol_var',       C['text']),
        ]:
            f = tk.Frame(body, bg=C['bg1']); f.pack(fill='x', pady=1)
            tk.Label(f, text=lbl, bg=C['bg1'], fg=C['text3'],
                     font=self.F_LABEL, width=17, anchor='w').pack(side='left')
            tk.Label(f, textvariable=getattr(self, vname),
                     bg=C['bg1'], fg=col, font=self.F_MONO_LG).pack(side='left')

    # ── TAB: ORDER FLOW ───────────────────────────────────────────────────────

    def _build_tab_orderflow(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  ORDER FLOW  ')
        col_l = tk.Frame(tab, bg=C['bg'], width=400)
        col_l.pack(side='left', fill='y', padx=2, pady=2)
        col_l.pack_propagate(False)
        col_r = tk.Frame(tab, bg=C['bg'])
        col_r.pack(side='left', fill='both', expand=True, padx=2, pady=2)

        pf, body = self._panel(col_l, 'CUMULATIVE VOLUME DELTA')
        pf.pack(fill='x', pady=(0, 2))
        self.cvd_var   = tk.StringVar(value='--')
        self.vbuy_var  = tk.StringVar(value='--')
        self.vsell_var = tk.StringVar(value='--')
        self.cvd_lbl   = tk.Label(body, textvariable=self.cvd_var,
                                   bg=C['bg1'], fg=C['text'],
                                   font=(self.MONO, 22, 'bold'))
        self.cvd_lbl.pack(anchor='w')
        row = tk.Frame(body, bg=C['bg1']); row.pack(fill='x', pady=2)
        for lbl, vname, col in [('BUY', 'vbuy_var', C['green']), ('SELL', 'vsell_var', C['red'])]:
            f = tk.Frame(row, bg=C['bg2'], padx=8, pady=4,
                         highlightthickness=1, highlightbackground=C['border'])
            f.pack(side='left', padx=(0, 4))
            tk.Label(f, text=lbl, bg=C['bg2'], fg=C['text3'], font=self.F_LABEL).pack()
            tk.Label(f, textvariable=getattr(self, vname), bg=C['bg2'],
                     fg=col, font=(self.MONO, 13, 'bold')).pack()
        self.cvd_bar_frame = tk.Frame(body, bg=C['bg3'], height=14); self.cvd_bar_frame.pack(fill='x', pady=3)
        self.cvd_bar = tk.Frame(self.cvd_bar_frame, bg=C['green'], height=14)
        self.cvd_bar.place(relx=0.5, y=0, relheight=1.0, relwidth=0.0, anchor='nw')
        self.cvd_sig_var = tk.StringVar(value='Awaiting data...')
        tk.Label(body, textvariable=self.cvd_sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM, wraplength=340).pack(anchor='w', pady=3)

        pf2, body2 = self._panel(col_l, 'FOOTPRINT CHART', 'DELTA')
        pf2.pack(fill='both', expand=True)
        self.fp_canvas = tk.Canvas(body2, bg=C['bg1'], bd=0, highlightthickness=0, height=160)
        self.fp_canvas.pack(fill='both', expand=True)

        pf3, body3 = self._panel(col_r, 'LIQUIDATIONS', 'REAL-TIME')
        pf3.pack(fill='x', pady=(0, 2))
        self.liq_total_var = tk.StringVar(value='$0 total')
        tk.Label(body3, textvariable=self.liq_total_var, bg=C['bg1'],
                 fg=C['red'], font=(self.MONO, 12, 'bold')).pack(anchor='w')
        self.liq_list_frame = tk.Frame(body3, bg=C['bg1']); self.liq_list_frame.pack(fill='x')

        pf5, body5 = self._panel(col_r, 'MICRO STATS — 30m')
        pf5.pack(fill='x')
        for lbl, vname, col in [('HIGH','h30','green'),('LOW','l30','red'),
                                  ('RANGE','r30','amber'),('BUY DOM %','buydom','text'),
                                  ('DOMINANT','dom','cyan')]:
            r = tk.Frame(body5, bg=C['bg1']); r.pack(fill='x', pady=1)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text3'],
                     font=self.F_LABEL, width=14, anchor='w').pack(side='left')
            v = tk.StringVar(value='--')
            setattr(self, vname + '_var', v)
            tk.Label(r, textvariable=v, bg=C['bg1'], fg=C[col],
                     font=self.F_MONO_MED, anchor='e').pack(side='right')

    # ── TAB: ANALYTICS ────────────────────────────────────────────────────────

    def _build_tab_analytics(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  ANALYTICS  ')
        col_l = tk.Frame(tab, bg=C['bg'], width=400)
        col_l.pack(side='left', fill='y', padx=2, pady=2)
        col_l.pack_propagate(False)
        col_r = tk.Frame(tab, bg=C['bg'])
        col_r.pack(side='left', fill='both', expand=True, padx=2, pady=2)

        pf, body = self._panel(col_l, 'ATR MULTI-TIMEFRAME', 'VOLATILITY')
        pf.pack(fill='x', pady=(0, 2))
        self.atr5m_var  = tk.StringVar(value='--')
        self.atr30m_var = tk.StringVar(value='--')
        self.atr4h_var  = tk.StringVar(value='--')
        self.atr_sig_var = tk.StringVar(value='Awaiting data...')
        grid = tk.Frame(body, bg=C['bg1']); grid.pack(fill='x', pady=3)
        for i, (tf, vname, col) in enumerate([('5m ATR', 'atr5m', C['green']),
                                               ('30m ATR', 'atr30m', C['cyan']),
                                               ('4h ATR', 'atr4h', C['blue'])]):
            f = tk.Frame(grid, bg=C['bg2'], padx=6, pady=5,
                         highlightthickness=1, highlightbackground=C['border'])
            f.grid(row=0, column=i, padx=1, sticky='ew')
            grid.columnconfigure(i, weight=1)
            tk.Label(f, text=tf, bg=C['bg2'], fg=C['text3'], font=self.F_LABEL).pack()
            tk.Label(f, textvariable=getattr(self, vname + '_var'), bg=C['bg2'],
                     fg=col, font=(self.MONO, 13, 'bold')).pack()
        tk.Label(body, textvariable=self.atr_sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM, wraplength=340).pack(anchor='w', pady=3)

        pf2, body2 = self._panel(col_l, 'TRAILING STOP CALC', 'AUTO')
        pf2.pack(fill='x', pady=(0, 2))
        for lbl, vname, col in [('3-BAR LOW STOP', 'ts3bar', C['red']),
                                  ('1.5× ATR STOP', 'tsatr15', C['red']),
                                  ('2× ATR STOP', 'tsatr2', C['amber']),
                                  ('SUGGESTED', 'tssug', C['cyan'])]:
            r = tk.Frame(body2, bg=C['bg1']); r.pack(fill='x', pady=1)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text3'], font=self.F_LABEL,
                     width=17, anchor='w').pack(side='left')
            v = tk.StringVar(value='--')
            setattr(self, vname + '_var', v)
            tk.Label(r, textvariable=v, bg=C['bg1'], fg=col,
                     font=(self.MONO, 12, 'bold')).pack(side='right')

        pf3, body3 = self._panel(col_l, 'POSITION PnL MONITOR', 'PAPER')
        pf3.pack(fill='both', expand=True)
        self.pos_status_var = tk.StringVar(value='No open position')
        self.pos_pnl_var    = tk.StringVar(value='--')
        tk.Label(body3, textvariable=self.pos_status_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_MED).pack(anchor='w')
        self.pos_pnl_lbl = tk.Label(body3, textvariable=self.pos_pnl_var,
                                     bg=C['bg1'], fg=C['text'],
                                     font=(self.MONO, 20, 'bold'))
        self.pos_pnl_lbl.pack(anchor='w')

        pf4, body4 = self._panel(col_r, 'DOMINANCE MATRIX — ALL PAIRS')
        pf4.pack(fill='both', expand=True)
        self._build_dm_table(body4, 'dm_rows')

    def _build_dm_table(self, parent, attr: str) -> None:
        headers = ['PAIR', 'PRICE', 'OBI', 'CVD', '1m%', '5m%', '15m%', 'SIGNAL']
        hdrs = tk.Frame(parent, bg=C['bg2']); hdrs.pack(fill='x')
        for h in headers:
            tk.Label(hdrs, text=h, bg=C['bg2'], fg=C['text3'],
                     font=self.F_HEAD, width=11, anchor='e', padx=4).pack(side='left')
        rows = {}
        for p in PAIRS:
            row = tk.Frame(parent, bg=C['bg1'],
                           highlightthickness=1, highlightbackground=C['border'])
            row.pack(fill='x', pady=2)
            cells = {}
            for h in headers:
                v = tk.StringVar(value='--')
                lbl = tk.Label(row, textvariable=v, bg=C['bg1'], fg=C['text'],
                               font=self.F_MONO_MED, width=11, anchor='e', padx=4)
                lbl.pack(side='left')
                cells[h] = (v, lbl)
            rows[p] = cells
        setattr(self, attr, rows)

    # ── TAB: RISK / CALC ──────────────────────────────────────────────────────

    def _build_tab_risk(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  RISK / CALC  ')
        col_l = tk.Frame(tab, bg=C['bg'], width=400)
        col_l.pack(side='left', fill='y', padx=2, pady=2)
        col_l.pack_propagate(False)
        col_r = tk.Frame(tab, bg=C['bg'])
        col_r.pack(side='left', fill='both', expand=True, padx=2, pady=2)

        pf, body = self._panel(col_l, 'KELLY CRITERION CALCULATOR')
        pf.pack(fill='x', pady=(0, 2))
        self.k_bal_var  = tk.StringVar(value='10000')
        self.k_wr_var   = tk.StringVar(value='55')
        self.k_wl_var   = tk.StringVar(value='2')
        self.k_size_var = tk.StringVar(value='--')
        self.k_pct_var  = tk.StringVar(value='--')
        self.k_atr_var  = tk.StringVar(value='--')
        for lbl, vname in [('Balance ($)', 'k_bal_var'),
                            ('Win Rate (%)', 'k_wr_var'),
                            ('Win/Loss Ratio', 'k_wl_var')]:
            r = tk.Frame(body, bg=C['bg1']); r.pack(fill='x', pady=2)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text2'], font=self.F_LABEL,
                     width=16, anchor='w').pack(side='left')
            e = tk.Entry(r, textvariable=getattr(self, vname),
                         bg=C['bg3'], fg=C['text'], font=self.F_MONO_MED,
                         insertbackground=C['cyan'], bd=0, width=12,
                         highlightthickness=1, highlightbackground=C['border2'])
            e.pack(side='right')
            e.bind('<KeyRelease>', lambda ev: self._calc_kelly())
        res = tk.Frame(body, bg=C['bg2'], padx=8, pady=6,
                        highlightthickness=1, highlightbackground=C['border2'])
        res.pack(fill='x', pady=4)
        tk.Label(res, text='POSITION SIZE', bg=C['bg2'], fg=C['text3'], font=self.F_HEAD).pack(anchor='w')
        tk.Label(res, textvariable=self.k_size_var, bg=C['bg2'],
                 fg=C['cyan'], font=(self.MONO, 20, 'bold')).pack(anchor='w')
        tk.Label(res, textvariable=self.k_pct_var, bg=C['bg2'],
                 fg=C['text2'], font=self.F_MONO_MED).pack(anchor='w')
        tk.Label(res, textvariable=self.k_atr_var, bg=C['bg2'],
                 fg=C['text3'], font=self.F_LABEL).pack(anchor='w')

        pf2, body2 = self._panel(col_l, 'RISK:REWARD CALCULATOR')
        pf2.pack(fill='x', pady=(0, 2))
        self.rr_entry_var = tk.StringVar(value='')
        self.rr_sl_var    = tk.StringVar(value='')
        self.rr_risk_var  = tk.StringVar(value='--')
        self.rr_tp_var    = tk.StringVar(value='--')
        self.rr_feas_var  = tk.StringVar(value='Awaiting ATR...')
        for lbl, vname in [('Entry Price', 'rr_entry_var'), ('Stop Loss', 'rr_sl_var')]:
            r = tk.Frame(body2, bg=C['bg1']); r.pack(fill='x', pady=2)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text2'], font=self.F_LABEL,
                     width=14, anchor='w').pack(side='left')
            e = tk.Entry(r, textvariable=getattr(self, vname),
                         bg=C['bg3'], fg=C['text'], font=self.F_MONO_MED,
                         insertbackground=C['cyan'], bd=0, width=14,
                         highlightthickness=1, highlightbackground=C['border2'])
            e.pack(side='right')
            e.bind('<KeyRelease>', lambda ev: self._calc_rr())
        for lbl, vname, col in [('RISK', 'rr_risk_var', C['red']),
                                  ('TARGET (2:1)', 'rr_tp_var', C['green']),
                                  ('ATR FEASIBILITY', 'rr_feas_var', C['text2'])]:
            r = tk.Frame(body2, bg=C['bg1']); r.pack(fill='x', pady=1)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text3'], font=self.F_LABEL,
                     width=16, anchor='w').pack(side='left')
            tk.Label(r, textvariable=getattr(self, vname), bg=C['bg1'],
                     fg=col, font=self.F_MONO_SM).pack(side='right')

        pf3, body3 = self._panel(col_r, 'EXECUTION MODULE', 'PAPER MODE')
        pf3.pack(fill='x', pady=(0, 2))
        self.exec_price_var  = tk.StringVar(value='--')
        self.exec_size_var   = tk.StringVar(value='--')
        self.exec_status_var = tk.StringVar(value='PAPER TRADING — NO REAL ORDERS SENT')
        for lbl, vname, col in [('PRICE', 'exec_price_var', C['green']),
                                  ('KELLY SIZE', 'exec_size_var', C['cyan'])]:
            r = tk.Frame(body3, bg=C['bg1']); r.pack(fill='x', pady=1)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text3'], font=self.F_LABEL,
                     width=12, anchor='w').pack(side='left')
            tk.Label(r, textvariable=getattr(self, vname), bg=C['bg1'],
                     fg=col, font=(self.MONO, 12, 'bold')).pack(side='right')
        btns = tk.Frame(body3, bg=C['bg1']); btns.pack(fill='x', pady=4)
        tk.Button(btns, text='▲  LONG  [PAPER]', bg='#001f0a', fg=C['green'],
                  font=(self.MONO, 12, 'bold'), bd=0, padx=10, pady=12, cursor='hand2',
                  highlightthickness=1, highlightbackground=C['green'],
                  command=lambda: self._exec_order('LONG')).pack(side='left', fill='x', expand=True, padx=(0, 3))
        tk.Button(btns, text='▼  SHORT  [PAPER]', bg='#1a0005', fg=C['red'],
                  font=(self.MONO, 12, 'bold'), bd=0, padx=10, pady=12, cursor='hand2',
                  highlightthickness=1, highlightbackground=C['red'],
                  command=lambda: self._exec_order('SHORT')).pack(side='left', fill='x', expand=True)
        tk.Button(body3, text='✕  CLOSE POSITION', bg='#1a1200', fg=C['amber'],
                  font=(self.MONO, 12, 'bold'), bd=0, pady=10, cursor='hand2',
                  highlightthickness=1, highlightbackground=C['amber'],
                  command=lambda: self._exec_order('CLOSE')).pack(fill='x', pady=(0, 3))
        tk.Label(body3, textvariable=self.exec_status_var, bg=C['bg1'],
                 fg=C['amber'], font=self.F_MONO_SM).pack(anchor='center', pady=4)

    # ── TAB: DOMINANCE ────────────────────────────────────────────────────────

    def _build_tab_dominance(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  DOMINANCE  ')
        pf, body = self._panel(tab, 'MULTI-PAIR DOMINANCE MATRIX + CORRELATION')
        pf.pack(fill='both', expand=True, padx=2, pady=2)
        tk.Label(body, text='  Live cross-pair momentum analysis.',
                 bg=C['bg1'], fg=C['text3'], font=self.F_MONO_SM).pack(anchor='w', pady=(0, 8))
        self._build_dm_table(body, 'dm2_rows')

    # ── TAB: ALERTS ───────────────────────────────────────────────────────────

    def _build_tab_alerts(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  ALERTS  ')
        pf, body = self._panel(tab, 'SMART ALERT SYSTEM', 'AUTO-DEDUP')
        pf.pack(fill='both', expand=True, padx=2, pady=2)
        self.alert_count_var = tk.StringVar(value='0 events')
        tk.Label(body, textvariable=self.alert_count_var, bg=C['bg1'],
                 fg=C['text2'], font=(self.MONO, 11, 'bold')).pack(anchor='w', pady=(0, 6))
        sb = tk.Scrollbar(body); sb.pack(side='right', fill='y')
        self.alert_text = tk.Text(body, bg=C['bg1'], fg=C['text'],
                                   font=self.F_MONO_MED, bd=0,
                                   yscrollcommand=sb.set, state='disabled')
        self.alert_text.pack(fill='both', expand=True)
        sb.config(command=self.alert_text.yview)
        for tag, col in [('POC', C['amber']), ('SPIKE', C['purple']),
                          ('ABSORB', C['cyan']), ('LIQ', C['red']),
                          ('EXEC', C['green']), ('green', C['green']),
                          ('red', C['red']), ('amber', C['amber'])]:
            self.alert_text.tag_configure(tag, foreground=col)

    # ── TAB: ACCOUNT ──────────────────────────────────────────────────────────

    def _build_tab_account(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  ACCOUNT  ')
        self._account_view = AccountView(tab, self)

    # ── TAB: AI PREDICTION ────────────────────────────────────────────────────

    def _build_tab_ai(self) -> None:
        tab = tk.Frame(self.nb, bg=C['bg'])
        self.nb.add(tab, text='  🤖 AI SIGNAL  ')
        # Scrollable container so content isn't clipped on small screens
        canvas = tk.Canvas(tab, bg=C['bg'], bd=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(canvas, bg=C['bg'])
        win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(win_id, width=canvas.winfo_width())
        inner.bind('<Configure>', _on_configure)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win_id, width=e.width))
        self._ai_view = AIPredictionView(inner, self)

    # ── EVENT CALLBACKS ───────────────────────────────────────────────────────

    def _on_ws_status(self, event) -> None:
        self._ws_status_var.set(str(event.payload or ''))

    def _on_alert(self, event) -> None:
        self._render_alerts()

    # ── MAIN UPDATE LOOP ─────────────────────────────────────────────────────

    def _update_loop(self) -> None:
        try:
            snap = STATE.snapshot()
            self._update_all(snap)
        except Exception:
            pass
        self.after(250, self._update_loop)

    def _update_all(self, s) -> None:
        p   = s.price
        dec = DECIMALS[s.pair]

        # Topbar
        self._ws_status_var.set(s.ws_status)
        fr = s.funding
        self._funding_var.set(f'{("+" if fr >= 0 else "")}{fr:.4f}%' if fr else '--')
        self._clock_var.set(datetime.datetime.utcnow().strftime('%H:%M:%S'))

        if not p:
            self.price_var.set('--')
            self.price_lbl.config(fg=C['text3'])
            return

        # Price flash animation
        new_price_str = fmt(p, dec)
        if new_price_str != self.price_var.get() and s.prev_price:
            self._price_flash_color = C['green'] if p >= s.prev_price else C['red']
            self._price_flash_steps = 6
            self._price_base_color  = self._price_flash_color
        self.price_var.set(new_price_str)
        if self._price_flash_steps > 0:
            self._price_flash_steps -= 1
            t_ratio = self._price_flash_steps / 6.0
            bright  = _hex_to_rgb(self._price_flash_color)
            dim     = _hex_to_rgb(C['text2'])
            r2 = int(bright[0] * t_ratio + dim[0] * (1 - t_ratio))
            g2 = int(bright[1] * t_ratio + dim[1] * (1 - t_ratio))
            b2 = int(bright[2] * t_ratio + dim[2] * (1 - t_ratio))
            self.price_lbl.config(fg=f'#{r2:02x}{g2:02x}{b2:02x}')
        else:
            self.price_lbl.config(fg=self._price_base_color or C['green'])

        pct_chg = (p - s.prev_price) / max(s.prev_price, 1) * 100 if s.prev_price else 0
        self.price_chg_var.set(f'{"+" if pct_chg >= 0 else ""}{pct_chg:.3f}%')
        if s.bid: self.bid_var.set(fmt(s.bid, dec))
        if s.ask: self.ask_var.set(fmt(s.ask, dec))
        if s.bid and s.ask:
            sp  = s.ask - s.bid
            spp = sp / max(s.bid, 1) * 100
            self.spread_var.set(f'{sp:.4f} ({spp:.4f}%)')

        # VWAP
        vwap, up1, up2, dn1, dn2 = get_vwap()
        if vwap:
            STATE.vwap_val = vwap
            self.vwap_var.set(fmt(vwap, dec))
            self.vwap_band_vars['+2σ'].set(fmt(up2, dec))
            self.vwap_band_vars['+1σ'].set(fmt(up1, dec))
            self.vwap_band_vars['DIST'].set(f'{(p - vwap)/vwap*100:.3f}%')
            self.vwap_band_vars['-1σ'].set(fmt(dn1, dec))
            self.vwap_band_vars['-2σ'].set(fmt(dn2, dec))
            self.vwap_sig_var.set('▲ ABOVE VWAP' if p > vwap else '▼ BELOW VWAP')

        # RSI (vectorised)
        prices_list = [x.price for x in list(s.atr5m_prices)[-30:]]
        if len(prices_list) > 15:
            STATE.rsi_1m = calc_rsi(prices_list)
            STATE.rsi_5m = calc_rsi(prices_list[::2] if len(prices_list) > 20 else prices_list)
        for rsi_val, name, smooth_attr in [
                (s.rsi_1m, 'rsi1m', '_rsi1m_smooth'),
                (s.rsi_5m, 'rsi5m', '_rsi5m_smooth')]:
            getattr(self, name + '_var').set(f'{rsi_val:.1f}')
            col = C['red'] if rsi_val > 70 else C['green'] if rsi_val < 30 else C['text']
            getattr(self, name + '_lbl').config(fg=col)
            smoothed = getattr(self, smooth_attr)
            smoothed = smoothed + (rsi_val - smoothed) * 0.25
            setattr(self, smooth_attr, smoothed)
            bar   = getattr(self, f'{name}_bar')
            frame = getattr(self, f'{name}_frame')
            frame.update_idletasks()
            w = frame.winfo_width()
            if w > 1:
                bar.place(x=0, y=0, relheight=1.0, width=int(w * smoothed / 100))
                bar.config(bg=col)
        rsi_sig = ('⚠ OVERBOUGHT — RSI > 70' if s.rsi_1m > 70 else
                   '⚠ OVERSOLD — RSI < 30'   if s.rsi_1m < 30 else 'Neutral RSI zone')
        self.rsi_sig_var.set(rsi_sig)

        # OI
        if s.oi:
            self.oi_var.set(fmt_oi(s.oi))
            h1   = [x for x in s.oi_history if now_ms() - x.t < 3_600_000]
            h4   = [x for x in s.oi_history if now_ms() - x.t < 14_400_000]
            chg1 = (s.oi - h1[0].oi) / max(h1[0].oi, 1) * 100 if len(h1) > 1 else 0
            chg4 = (s.oi - h4[0].oi) / max(h4[0].oi, 1) * 100 if len(h4) > 1 else 0
            self.oi_chg1h_var.set(f'{"+" if chg1 >= 0 else ""}{chg1:.2f}%')
            self.oi_chg4h_var.set(f'{"+" if chg4 >= 0 else ""}{chg4:.2f}%')
            sig, _ = oi_signal()
            self.oi_sig_var.set(sig)

        # Tape
        self._update_tape(s)

        # Order book (object-pooled view)
        self._ob_view.update(s)

        # VPVR (object-pooled view)
        self._vpvr_view.update(s)

        # Funding
        col = (C['red'] if fr > 0.05 else C['amber'] if fr > 0.01 else
               C['green'] if fr < -0.05 else C['teal'] if fr < -0.01 else C['text2'])
        self.fund_rate_var.set(f'{"+" if fr >= 0 else ""}{fr:.4f}%' if fr else '--')
        self.fund_rate_lbl.config(fg=col)
        bias = ('LONGS OVERCROWDED' if fr > 0.05 else 'Longs paying' if fr > 0.01 else
                'SHORTS OVERCROWDED' if fr < -0.05 else 'Shorts paying' if fr < -0.01 else 'Neutral')
        self.fund_bias_var.set(bias)
        STATE.funding_cd = max(0, STATE.funding_cd - 250)
        cd = STATE.funding_cd
        self.fund_cd_var.set(f'{cd//3600000:02d}:{(cd%3600000)//60000:02d}:{(cd%60000)//1000:02d}')

        # Context
        regime, _ = market_regime(s.pair)
        comp, _   = composite_signal()
        self.ctx_regime_var.set(regime)
        self.ctx_composite_var.set(comp)
        adr     = s.adr_high - s.adr_low if s.adr_high and s.adr_low != float('inf') else 0
        adr_used = abs(p - s.adr_low) / max(adr, 1) * 100 if adr and p else 0
        self.ctx_adr_var.set(f'ADR: {adr_used:.1f}%')
        atr5  = calc_atr(s.pair, '5m')
        atr30 = calc_atr(s.pair, '30m')
        vol_r = atr5 / max(atr30, 0.001) if atr5 and atr30 else 0
        self.ctx_vol_var.set(f'Vol: {vol_r:.2f}x {"⚡" if vol_r > 1.5 else ""}')

        # CVD tab
        cvd = s.cvd
        self.cvd_var.set(('+' if cvd >= 0 else '') + fmt_k(abs(cvd)))
        self.cvd_lbl.config(fg=C['green'] if cvd >= 0 else C['red'])
        self.vbuy_var.set(fmt_k(s.vbuy))
        self.vsell_var.set(fmt_k(s.vsell))
        tot = s.vbuy + s.vsell or 1
        pct_target = 0.5 + cvd / max(abs(cvd), tot) * 0.5
        self._cvd_pct_smooth += (pct_target - self._cvd_pct_smooth) * 0.2
        pct = self._cvd_pct_smooth
        self.cvd_bar_frame.update_idletasks()
        w = self.cvd_bar_frame.winfo_width()
        if w > 1:
            if pct >= 0.5:
                self.cvd_bar.place(x=w//2, y=0, relheight=1.0,
                                    width=max(1, int(w * (pct - 0.5))), anchor='nw')
                self.cvd_bar.config(bg=C['green'])
            else:
                bw2 = max(1, int(w * (0.5 - pct)))
                self.cvd_bar.place(x=w//2 - bw2, y=0, relheight=1.0,
                                    width=bw2, anchor='nw')
                self.cvd_bar.config(bg=C['red'])

        if len(s.atr_prices) > 10:
            pl = [x.price for x in s.atr_prices]
            p_up = pl[-1] > pl[-10]
            if   p_up and cvd < 0:  self.cvd_sig_var.set('⚠ BEARISH DIV')
            elif not p_up and cvd>0:self.cvd_sig_var.set('⚠ BULLISH DIV')
            elif cvd > 0:           self.cvd_sig_var.set('Bullish flow dominant')
            else:                   self.cvd_sig_var.set('Bearish flow dominant')

        # Micro stats
        if s.atr_prices:
            pl2 = [x.price for x in s.atr_prices]
            h2 = max(pl2); l2 = min(pl2)
            self.h30_var.set(fmt(h2, dec)); self.l30_var.set(fmt(l2, dec))
            self.r30_var.set(fmt(h2 - l2, dec))
            bp = s.vbuy / (s.vbuy + s.vsell + 1) * 100
            self.buydom_var.set(f'{bp:.1f}%')
            self.dom_var.set('BUYERS' if bp > 55 else 'SELLERS' if bp < 45 else 'NEUTRAL')

        self._update_footprint(s)
        self._update_atr(s)
        self._update_trailing_stop(s)
        self._update_position(s)
        self._update_dominance(s)
        self._render_alerts()

        self.exec_price_var.set(fmt(p, dec) if p else '--')
        self._calc_kelly()

    def _update_tape(self, s) -> None:
        trades = list(s.trades)[:20]
        tape_sig = tuple((float(t.price), t.is_buy, t.tier, t.ts) for t in trades)
        tps = len(s.trade_win) / 5.0
        self.tps_var.set(f'{tps:.1f} t/s  {"🔥" if tps > 20 else ""}')
        if tape_sig == self._tape_hash:
            return
        self._tape_hash = tape_sig
        self.tape_list.config(state='normal')
        self.tape_list.delete('1.0', 'end')
        dec = DECIMALS[s.pair]
        for t in trades:
            side_tag = 'buy' if t.is_buy else 'sell'
            tier_tag = t.tier.lower() if t.tier in ('WHALE', 'INST', 'DOLPH') else ''
            self.tape_list.insert('end', t.ts + '  ', 'ts')
            self.tape_list.insert('end', fmt(float(t.price), dec).rjust(12) + '  ', side_tag)
            self.tape_list.insert('end', f'{float(t.qty):.3f}'.rjust(8) + '  ', 'vol')
            self.tape_list.insert('end', ('BUY' if t.is_buy else 'SELL').ljust(5), side_tag)
            if tier_tag:
                self.tape_list.insert('end', t.tier, tier_tag)
            self.tape_list.insert('end', '\n')
        self.tape_list.config(state='disabled')

    def _update_footprint(self, s) -> None:
        if not s.footprint or not s.price:
            self.fp_canvas.delete('all')
            return
        nearby = sorted([b for b in s.footprint if abs(b - s.price) / s.price < 0.005],
                         reverse=True)[:8]
        if not nearby:
            self.fp_canvas.delete('all')
            return
        cv = self.fp_canvas; cv.update_idletasks()
        cw = cv.winfo_width() or 340; row_h = 18; label_w = 55
        half = (cw - label_w - 10) // 2
        cv.config(height=len(nearby) * row_h + 4)
        cv.delete('all')
        font_tiny = self.F_TINY
        dec = DECIMALS[s.pair]
        threshold = 0.5 if s.pair == 'SOLUSDT' else 5 if s.pair == 'ETHUSDT' else 50
        max_b = max(s.footprint[b]['buy']  for b in nearby) or 1
        max_s = max(s.footprint[b]['sell'] for b in nearby) or 1
        y = 2
        for b in nearby:
            fp     = s.footprint[b]
            delta  = fp['buy'] - fp['sell']
            is_cur = abs(b - s.price) < threshold
            cv.create_rectangle(0, y, cw, y + row_h - 1,
                                  fill=C['bg3'] if is_cur else C['bg1'], outline='')
            col_p = C['cyan'] if is_cur else (C['red'] if b > s.price else C['green'])
            cv.create_text(label_w - 2, y + row_h // 2, text=fmt(b, 0),
                            fill=col_p, font=font_tiny, anchor='e')
            bw = max(3, int(fp['buy'] / max_b * half))
            cv.create_rectangle(label_w, y + 3, label_w + bw, y + row_h - 3,
                                  fill='#003300', outline='')
            cv.create_text(label_w + bw + 2, y + row_h // 2,
                            text=fmt_k(fp['buy']).replace('$', ''),
                            fill=C['green'], font=font_tiny, anchor='w')
            sw  = max(3, int(fp['sell'] / max_s * half))
            sx  = label_w + half + 10
            cv.create_rectangle(sx, y + 3, sx + sw, y + row_h - 3,
                                  fill='#330000', outline='')
            cv.create_text(sx + sw + 2, y + row_h // 2,
                            text=fmt_k(fp['sell']).replace('$', ''),
                            fill=C['red'], font=font_tiny, anchor='w')
            dc = C['green'] if delta > 0 else C['red']
            cv.create_text(cw - 2, y + row_h // 2,
                            text=('+' if delta > 0 else '') + fmt_k(abs(delta)).replace('$', ''),
                            fill=dc, font=font_tiny, anchor='e')
            y += row_h

    def _update_atr(self, s) -> None:
        atr5  = calc_atr(s.pair, '5m')
        atr30 = calc_atr(s.pair, '30m')
        atr4h = atr30 * 3.5 if atr30 else 0
        self.atr5m_var.set(fmt(atr5, 2)  if atr5  else '--')
        self.atr30m_var.set(fmt(atr30, 2) if atr30 else '--')
        self.atr4h_var.set(fmt(atr4h, 2) if atr4h else '--')
        if atr5 and atr30:
            ratio = atr5 / max(atr30, 0.001)
            sig   = ('⚡ VOL EXPLOSION — 5m ATR > 1.5× baseline' if ratio > 1.5 else
                     '↑ Vol increasing — 5m ATR elevated'          if ratio > 1.1 else
                     'Compression — 5m ATR low. Coiling.'           if ratio < 0.5 else
                     'Normal volatility regime.')
            self.atr_sig_var.set(sig)

    def _update_trailing_stop(self, s) -> None:
        p = s.price
        if not p: return
        atr30  = calc_atr(s.pair, '30m')
        recent = [x.price for x in s.atr_prices if now_ms() - x.t < 900_000]
        lo3    = min(recent) if recent else p
        atr15  = p - atr30 * 1.5 if atr30 else 0
        atr2   = p - atr30 * 2   if atr30 else 0
        sug    = max(atr15, lo3) if atr15 and lo3 else (atr15 or lo3)
        dec    = DECIMALS[s.pair]
        self.ts3bar_var.set(fmt(lo3,  dec) if lo3   else '--')
        self.tsatr15_var.set(fmt(atr15, dec) if atr15 else '--')
        self.tsatr2_var.set(fmt(atr2,  dec) if atr2  else '--')
        self.tssug_var.set(fmt(sug,   dec) if sug   else '--')

    def _update_position(self, s) -> None:
        if not s.position:
            self.pos_status_var.set('No open position')
            self.pos_pnl_var.set('--')
            return
        pos = s.position
        pnl = pos.pnl(Decimal(str(s.price)))
        self.pos_status_var.set(
            f'{pos.side} @ {fmt(float(pos.entry), DECIMALS[s.pair])} | ${float(pos.size):.0f}')
        self.pos_pnl_var.set(f'{"+" if pnl >= 0 else ""}${float(pnl):.2f}')
        self.pos_pnl_lbl.config(fg=C['green'] if pnl >= 0 else C['red'])

    def _update_dominance(self, s) -> None:
        for p in PAIRS:
            price = s.dm_prices.get(p, 0)
            obi   = s.dm_obi.get(p, 0)
            cvd   = s.dm_cvd.get(p, 0)
            now   = now_ms()
            r1  = [x for x in s.corr_prices[p] if now - x.t < 60_000]
            r5  = [x for x in s.corr_prices[p] if now - x.t < 300_000]
            r15 = [x for x in s.corr_prices[p] if now - x.t < 900_000]
            p1  = (r1[-1].price  - r1[0].price)  / r1[0].price  * 100 if len(r1)  > 1 else 0
            p5  = (r5[-1].price  - r5[0].price)  / r5[0].price  * 100 if len(r5)  > 1 else 0
            p15 = (r15[-1].price - r15[0].price) / r15[0].price * 100 if len(r15) > 1 else 0
            mom = sum([1 if p15 > 0.1 else -1 if p15 < -0.1 else 0,
                       1 if p5  > 0.05 else -1 if p5 < -0.05 else 0,
                       1 if obi > 0.2  else -1 if obi < -0.2  else 0,
                       1 if cvd > 0    else -1])
            sig     = ('LONG ▲' if mom >= 3 else 'SHORT ▼' if mom <= -3 else
                       'BIAS↑' if mom >= 1 else 'BIAS↓' if mom <= -1 else 'NEUTRAL')
            sig_col = (C['green'] if mom >= 3 else C['red'] if mom <= -3 else
                       C['teal'] if mom >= 1 else C['amber'] if mom <= -1 else C['text3'])
            dec_p   = DECIMALS[p]
            for table in [getattr(self, 'dm_rows', {}), getattr(self, 'dm2_rows', {})]:
                if p not in table: continue
                cells = table[p]
                bg_c  = C['bg3'] if p == s.pair else C['bg1']
                for h, val, col in [
                    ('PAIR',  PAIR_LABELS[p],                     C['cyan'] if p == s.pair else C['text']),
                    ('PRICE', fmt(price, dec_p) if price else '--', C['text']),
                    ('OBI',   f'{obi:+.3f}',                       C['green'] if obi > 0.2 else C['red'] if obi < -0.2 else C['text3']),
                    ('CVD',   '▲' if cvd > 0 else '▼',            C['green'] if cvd > 0 else C['red']),
                    ('1m%',   f'{p1:+.3f}%',                       C['green'] if p1 > 0 else C['red']),
                    ('5m%',   f'{p5:+.3f}%',                       C['green'] if p5 > 0 else C['red']),
                    ('15m%',  f'{p15:+.3f}%',                      C['green'] if p15 > 0 else C['red']),
                    ('SIGNAL',sig,                                  sig_col),
                ]:
                    if h not in cells: continue
                    v_var, lbl = cells[h]
                    v_var.set(val)
                    lbl.config(fg=col, bg=bg_c)

    def _render_alerts(self) -> None:
        alerts = list(STATE.alerts)
        sig    = tuple((a.type, a.msg, a.ts) for a in alerts[:10])
        if sig == self._alert_hash:
            return
        self._alert_hash = sig
        self.alert_count_var.set(f'{STATE.alert_count} events')
        self.alert_text.config(state='normal')
        self.alert_text.delete('1.0', 'end')
        for a in alerts:
            self.alert_text.insert('end', f'[{a.ts}] ', 'ts' if hasattr(self.alert_text, 'ts') else 'end')
            self.alert_text.insert('end', f'[{a.type}] ', a.type)
            self.alert_text.insert('end', a.msg + '\n', a.color)
        self.alert_text.config(state='disabled')

    # ── CONTROLS ─────────────────────────────────────────────────────────────

    def _switch_pair(self, pair: str) -> None:
        if pair == STATE.pair:
            return
        for p, b in self._pair_btns.items():
            b.config(bg=C['cyan'] if p == pair else C['bg2'],
                     fg='#000' if p == pair else C['text2'])
        STATE.tape_frozen = False
        self.tape_status_var.set('LIVE')
        self._feed.switch_pair(pair)
        # Re-attach AI engine to new feed loop after short delay
        self.after(300, self._feed.start_ai)

    def _toggle_freeze(self) -> None:
        STATE.tape_frozen = not STATE.tape_frozen
        if not STATE.tape_frozen:
            while STATE.tape_buffer:
                STATE.trades.appendleft(STATE.tape_buffer.popleft())
        self.tape_status_var.set('FROZEN ❄' if STATE.tape_frozen else 'LIVE ▶')

    def _exec_order(self, side: str) -> None:
        p = STATE.price
        if not p:
            self.exec_status_var.set('NO PRICE — CANNOT PLACE ORDER')
            return
        try:
            bal = Decimal(self.k_bal_var.get())
            wr  = float(self.k_wr_var.get()) / 100
            wl  = float(self.k_wl_var.get())
            size, _, _ = calc_kelly(bal, wr, wl)
        except Exception:
            size = Decimal('0')

        if side == 'CLOSE':
            if not STATE.position:
                self.exec_status_var.set('NO OPEN POSITION')
                return
            pnl = STATE.position.pnl(Decimal(str(p)))
            self.exec_status_var.set(
                f'CLOSED {STATE.position.side} @ {fmt(p)} | PnL: '
                f'{"+" if pnl >= 0 else ""}${float(pnl):.2f}')
            add_alert('EXEC', f'Paper CLOSE @ {fmt(p)} PnL: ${float(pnl):.2f}',
                      'green' if pnl >= 0 else 'red')
            STATE.position = None
            return

        if STATE.position:
            self.exec_status_var.set('CLOSE EXISTING POSITION FIRST')
            return

        from models import Position
        STATE.position = Position(
            side=side, entry=Decimal(str(p)),
            size=size, t=now_ms()
        )
        self.exec_status_var.set(
            f'PAPER {side} ${float(size):.0f} @ {fmt(p, DECIMALS[STATE.pair])}')
        add_alert('EXEC', f'Paper {side} @ {fmt(p)} sz ${float(size):.0f}', 'amber')

    def _calc_kelly(self) -> None:
        try:
            bal = Decimal(self.k_bal_var.get())
            wr  = float(self.k_wr_var.get()) / 100
            wl  = float(self.k_wl_var.get())
            pos, f, hk = calc_kelly(bal, wr, wl)
            self.k_size_var.set(f'${float(pos):.2f}')
            self.k_pct_var.set(f'f*={f*100:.2f}% | HK={hk*100:.2f}%')
            self.exec_size_var.set(f'${float(pos):.2f}')
        except Exception:
            pass

    def _calc_rr(self) -> None:
        try:
            entry = float(self.rr_entry_var.get())
            sl    = float(self.rr_sl_var.get())
            risk  = abs(entry - sl)
            tp    = entry + risk * 2
            self.rr_risk_var.set(f'${risk:.2f}')
            self.rr_tp_var.set(f'${tp:.2f}')
            atr = calc_atr(STATE.pair)
            if atr:
                d = tp - entry
                if   d <= atr * 0.5: self.rr_feas_var.set('✓ VIABLE — inside 0.5×ATR')
                elif d <= atr:        self.rr_feas_var.set('~ POSSIBLE — near 1×ATR')
                else:                 self.rr_feas_var.set('✗ UNREALISTIC — exceeds ATR')
        except Exception:
            pass


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Warn if async libs missing (will fall back to sim)
    try:
        import aiohttp, websockets
    except ImportError:
        print('Note: aiohttp/websockets not installed. Running in SIM mode.')
        print('For live data: pip install aiohttp websockets python-dotenv numpy')
        print()

    app = HPAT_App()

    def on_key(e: tk.Event) -> None:
        k = e.char.upper() if e.char else ''
        key_map = {'1': 'BTCUSDT', '2': 'ETHUSDT', '3': 'SOLUSDT',
                   '4': 'BNBUSDT', '5': 'XRPUSDT'}
        if k in key_map:
            app._switch_pair(key_map[k])
        elif k == 'F':
            app._toggle_freeze()
        elif k == 'B':
            app._exec_order('LONG')
        elif k == 'S':
            app._exec_order('SHORT')
        elif k == 'X':
            app._exec_order('CLOSE')

    app.bind('<Key>', on_key)
    app.mainloop()
