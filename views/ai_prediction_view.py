"""
HPAT v6 — views/ai_prediction_view.py
AI Prediction Panel powered by Gemini.

UI layout:
  ┌─ GEMINI AI SIGNAL ENGINE ────────────────────────────┐
  │  [API KEY ENTRY]  [ENABLE]        Status: LIVE  10s  │
  ├──────────────────────────────────────────────────────┤
  │  DIRECTION   CONVICTION   HORIZON   RISK   SIZE%     │
  │  ▲ LONG          8/10     30-60s   MEDIUM  1.2%      │
  ├──────────────────────────────────────────────────────┤
  │  ENTRY ZONE   SL        TP1       TP2     R:R        │
  │  65100-65200  64800     65500     66000   2.5        │
  ├──────────────────────────────────────────────────────┤
  │  PRIMARY DRIVER                                       │
  │  CVD divergence resolving bullish with OBI +0.42...  │
  ├──────────────────────────────────────────────────────┤
  │  CONFLUENCE                    INVALIDATION           │
  │  • RSI 1m oversold bounce      Close < 64900          │
  │  • OBI bid-heavy +0.42                                │
  │  • Price reclaimed POC                                │
  ├──────────────────────────────────────────────────────┤
  │  REASONING CHAIN                                      │
  │  [collapsible text widget]                            │
  ├──────────────────────────────────────────────────────┤
  │  CALL HISTORY (last 10)                               │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Deque
import collections
import datetime

from views.base_view import BaseView
from models import C, BUS


class AIPredictionView(BaseView):

    MAX_HISTORY = 20

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()
        self._history: Deque = collections.deque(maxlen=self.MAX_HISTORY)
        self._countdown   = 10
        self._enabled     = False
        self._last_dir    = 'NO_EDGE'
        self._blink_on    = True

        self._build_ui(parent)

        # Subscribe to EventBus — callback fires on feed thread; we schedule to Tk
        self.subscribe('ai_prediction',       lambda e: self._schedule(self._on_prediction, e.payload))
        self.subscribe('ai_prediction_error', lambda e: self._schedule(self._on_error, e.payload))

        # Countdown ticker (every 1 s from Tk main thread)
        self._tick()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self, parent: tk.Widget) -> None:
        C = self._C

        # ── HEADER / CONFIG ──────────────────────────────────────────────────
        pf, hdr_body = self._panel(parent, '🤖 GEMINI AI SIGNAL ENGINE v2.0',
                                    right_text='10s PULSE', accent=C['purple'])
        pf.pack(fill='x', padx=4, pady=(4, 2))

        config_row = tk.Frame(hdr_body, bg=C['bg1'])
        config_row.pack(fill='x', pady=(0, 4))

        tk.Label(config_row, text='API KEY:', bg=C['bg1'],
                 fg=C['text3'], font=self.F_LABEL).pack(side='left', padx=(0, 4))

        self._key_var = tk.StringVar()
        # Pre-fill from env if available
        self._key_var.set(os.environ.get('GEMINI_API_KEY', ''))
        key_entry = tk.Entry(config_row, textvariable=self._key_var,
                             show='•', width=36,
                             bg=C['bg3'], fg=C['text'],
                             font=self.F_MONO_SM, bd=0, insertbackground=C['purple'],
                             highlightthickness=1, highlightbackground=C['border2'])
        key_entry.pack(side='left', padx=(0, 6))

        self._toggle_btn = tk.Button(
            config_row, text='▶ ENABLE',
            bg=C['bg3'], fg=C['purple'],
            font=(self.MONO, 10, 'bold'),
            bd=0, padx=10, pady=4, cursor='hand2',
            highlightthickness=1, highlightbackground=C['purple'],
            command=self._toggle_engine,
        )
        self._toggle_btn.pack(side='left', padx=(0, 8))

        # Status pill
        status_frame = tk.Frame(config_row, bg=C['bg2'], padx=8, pady=3,
                                 highlightthickness=1, highlightbackground=C['border'])
        status_frame.pack(side='left')
        self._status_dot = tk.Label(status_frame, text='●', bg=C['bg2'],
                                     fg=C['text3'], font=self.F_TINY)
        self._status_dot.pack(side='left')
        self._status_var = tk.StringVar(value='OFFLINE')
        tk.Label(status_frame, textvariable=self._status_var,
                 bg=C['bg2'], fg=C['text3'],
                 font=(self.MONO, 10, 'bold')).pack(side='left', padx=(4, 0))

        # Countdown + latency
        right_info = tk.Frame(config_row, bg=C['bg1'])
        right_info.pack(side='right')
        self._countdown_var = tk.StringVar(value='NEXT: --s')
        self._latency_var   = tk.StringVar(value='')
        tk.Label(right_info, textvariable=self._countdown_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='right', padx=4)
        tk.Label(right_info, textvariable=self._latency_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='right')

        # ── MAIN SIGNAL CARD ─────────────────────────────────────────────────
        self._card = tk.Frame(hdr_body, bg=C['bg2'],
                               highlightthickness=1, highlightbackground=C['border2'])
        self._card.pack(fill='x', pady=(2, 4))

        # Row 1: Direction + metrics
        top_row = tk.Frame(self._card, bg=C['bg2'])
        top_row.pack(fill='x', padx=10, pady=(8, 4))

        # Big direction label
        self._dir_var = tk.StringVar(value='── NO SIGNAL ──')
        self._dir_lbl = tk.Label(top_row, textvariable=self._dir_var,
                                  bg=C['bg2'], fg=C['text3'],
                                  font=(self.MONO, 22, 'bold'))
        self._dir_lbl.pack(side='left')

        # Conviction gauge
        self._conv_frame = tk.Frame(top_row, bg=C['bg2'])
        self._conv_frame.pack(side='left', padx=16)
        tk.Label(self._conv_frame, text='CONVICTION', bg=C['bg2'],
                 fg=C['text3'], font=self.F_TINY).pack()
        self._conv_var = tk.StringVar(value='--/10')
        self._conv_lbl = tk.Label(self._conv_frame, textvariable=self._conv_var,
                                   bg=C['bg2'], fg=C['text3'],
                                   font=(self.MONO, 18, 'bold'))
        self._conv_lbl.pack()
        # Conviction bar (10 blocks)
        self._conv_bar_frame = tk.Frame(self._conv_frame, bg=C['bg3'], height=6, width=100)
        self._conv_bar_frame.pack(fill='x', pady=2)
        self._conv_bar = tk.Frame(self._conv_bar_frame, bg=C['text3'], height=6)
        self._conv_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # Metric cells
        for label, vattr, color in [
            ('HORIZON',  '_horizon_var', C['text2']),
            ('RISK',     '_risk_var',    C['amber']),
            ('SIZE %',   '_size_var',    C['cyan']),
            ('R:R',      '_rr_var',      C['teal']),
        ]:
            f = tk.Frame(top_row, bg=C['bg3'], padx=10, pady=4,
                         highlightthickness=1, highlightbackground=C['border'])
            f.pack(side='left', padx=(0, 4))
            tk.Label(f, text=label, bg=C['bg3'], fg=C['text3'], font=self.F_TINY).pack()
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(f, textvariable=v, bg=C['bg3'], fg=color,
                     font=(self.MONO, 12, 'bold')).pack()

        # Row 2: Price levels
        levels_row = tk.Frame(self._card, bg=C['bg2'])
        levels_row.pack(fill='x', padx=10, pady=(0, 8))
        for label, vattr, color in [
            ('ENTRY ZONE', '_entry_var', C['text']),
            ('STOP LOSS',  '_sl_var',    C['red']),
            ('TP1',        '_tp1_var',   C['green']),
            ('TP2',        '_tp2_var',   C['teal']),
        ]:
            f = tk.Frame(levels_row, bg=C['bg3'], padx=8, pady=4,
                         highlightthickness=1, highlightbackground=C['border'])
            f.pack(side='left', padx=(0, 4))
            tk.Label(f, text=label, bg=C['bg3'], fg=C['text3'], font=self.F_TINY).pack(anchor='w')
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(f, textvariable=v, bg=C['bg3'], fg=color,
                     font=(self.MONO, 11, 'bold')).pack(anchor='w')

        # ── PRIMARY DRIVER ────────────────────────────────────────────────────
        pf2, body2 = self._panel(hdr_body, 'PRIMARY DRIVER')
        pf2.pack(fill='x', pady=(0, 2))
        self._driver_var = tk.StringVar(value='Waiting for first prediction...')
        tk.Label(body2, textvariable=self._driver_var,
                 bg=C['bg1'], fg=C['text2'],
                 font=(self.MONO, 11, 'bold'),
                 wraplength=900, justify='left').pack(anchor='w')

        # ── CONFLUENCE + INVALIDATION (side by side) ─────────────────────────
        cf_row = tk.Frame(hdr_body, bg=C['bg'])
        cf_row.pack(fill='x', pady=(0, 2))

        pf3, body3 = self._panel(cf_row, 'CONFLUENCE FACTORS')
        pf3.pack(side='left', fill='both', expand=True, padx=(0, 2))
        self._confluence_text = tk.Text(body3, bg=C['bg1'], fg=C['teal'],
                                         font=self.F_MONO_SM, bd=0, height=4,
                                         state='disabled', wrap='word',
                                         selectbackground=C['bg3'])
        self._confluence_text.pack(fill='both', expand=True)

        pf4, body4 = self._panel(cf_row, 'INVALIDATION SCENARIO', accent=C['red'])
        pf4.pack(side='left', fill='both', expand=True)
        self._inval_var = tk.StringVar(value='--')
        tk.Label(body4, textvariable=self._inval_var,
                 bg=C['bg1'], fg=C['red'],
                 font=(self.MONO, 10, 'bold'),
                 wraplength=350, justify='left').pack(anchor='w')
        self._regime_var = tk.StringVar(value='--')
        tk.Label(body4, textvariable=self._regime_var,
                 bg=C['bg1'], fg=C['blue'],
                 font=self.F_MONO_SM).pack(anchor='w', pady=(4, 0))

        # ── REASONING CHAIN ───────────────────────────────────────────────────
        pf5, body5 = self._panel(hdr_body, 'AI REASONING CHAIN', accent=C['blue'])
        pf5.pack(fill='x', pady=(0, 2))
        rsb = tk.Scrollbar(body5, bg=C['bg2'], troughcolor=C['bg3'])
        rsb.pack(side='right', fill='y')
        self._reasoning_text = tk.Text(body5, bg=C['bg1'], fg=C['text2'],
                                        font=self.F_MONO_SM, bd=0, height=5,
                                        yscrollcommand=rsb.set,
                                        state='disabled', wrap='word',
                                        selectbackground=C['bg3'])
        self._reasoning_text.pack(fill='both', expand=True)
        rsb.config(command=self._reasoning_text.yview)
        for tag, col in [('step', C['text3']), ('text', C['text2'])]:
            self._reasoning_text.tag_configure(tag, foreground=col)

        # ── PREDICTION HISTORY ────────────────────────────────────────────────
        pf6, body6 = self._panel(hdr_body, f'SIGNAL HISTORY (last {self.MAX_HISTORY})')
        pf6.pack(fill='both', expand=True, pady=(0, 2))
        hsb = tk.Scrollbar(body6, bg=C['bg2'], troughcolor=C['bg3'])
        hsb.pack(side='right', fill='y')
        self._history_text = tk.Text(body6, bg=C['bg1'], fg=C['text'],
                                      font=self.F_MONO_SM, bd=0, height=8,
                                      yscrollcommand=hsb.set,
                                      state='disabled', wrap='none',
                                      selectbackground=C['bg3'])
        self._history_text.pack(fill='both', expand=True)
        hsb.config(command=self._history_text.yview)
        for tag, col in [('long', C['green']), ('short', C['red']),
                          ('neutral', C['text3']), ('no_edge', C['text3']),
                          ('ts', C['text3']), ('conv', C['amber']),
                          ('driver', C['text2']), ('err', C['red'])]:
            self._history_text.tag_configure(tag, foreground=col)

        # ── STATS FOOTER ──────────────────────────────────────────────────────
        footer = tk.Frame(hdr_body, bg=C['bg1'])
        footer.pack(fill='x')
        self._stats_var = tk.StringVar(value='Calls: 0  |  Errors: 0  |  Model: gemini-2.0-flash')
        tk.Label(footer, textvariable=self._stats_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='left')

    # ── Engine toggle ─────────────────────────────────────────────────────────

    def _toggle_engine(self) -> None:
        from gemini_engine import GEMINI
        C = self._C
        if self._enabled:
            # Disable
            GEMINI.enabled = False
            self._enabled  = False
            self._toggle_btn.config(text='▶ ENABLE', fg=C['purple'])
            self._set_status('OFFLINE', C['text3'])
            self._countdown_var.set('NEXT: --s')
        else:
            key = self._key_var.get().strip()
            if not key:
                self._set_status('NO KEY', C['red'])
                return
            ok = GEMINI.configure(key)
            if ok:
                self._enabled = True
                self._toggle_btn.config(text='■ DISABLE', fg=C['red'])
                self._set_status('LIVE ●', C['purple'])
                self._countdown = 10
            else:
                self._set_status('AUTH ERROR', C['red'])

    def _set_status(self, text: str, color: str) -> None:
        self._status_var.set(text)
        self._status_dot.config(fg=color)

    # ── Countdown ticker ─────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._enabled:
            self._countdown = max(0, self._countdown - 1)
            self._countdown_var.set(f'NEXT: {self._countdown}s')
            if self._countdown == 0:
                self._countdown = 10
            # Blink status dot when active
            self._blink_on = not self._blink_on
            self._status_dot.config(fg=self._C['purple'] if self._blink_on else self._C['bg2'])
        self._app.after(1000, self._tick)

    # ── Prediction handler ────────────────────────────────────────────────────

    def _on_prediction(self, result) -> None:
        """Called on Tk main thread via _schedule()."""
        from gemini_engine import GEMINI
        C = self._C

        # Only push to history and fully update card for real API calls
        if not result.skipped:
            self._history.appendleft(result)
            self._update_card(result)
            self._update_confluence(result)
            self._update_reasoning(result)
            self._update_history()
        else:
            # Skipped: just blink the status to show engine is alive
            skip_txt = f'CACHED ({result.skip_reason})'
            self._set_status(skip_txt, C['text3'])
            self._app.after(1500, lambda: self._set_status('LIVE ●', C['purple']))

        self._update_stats(GEMINI)
        # Reset countdown using actual cadence
        from gemini_engine import _CADENCE
        regime  = result.regime if result.regime in _CADENCE else 'RANGING'
        self._countdown = _CADENCE.get(regime, 30)

    def _on_error(self, msg: str) -> None:
        C = self._C
        self._set_status(f'ERR: {str(msg)[:40]}', C['red'])

    # ── Card update ──────────────────────────────────────────────────────────

    def _update_card(self, r) -> None:
        C   = self._C
        dec = 2  # DECIMALS handled dynamically

        # Direction
        dir_map = {
            'LONG':    ('▲  LONG',  C['green']),
            'SHORT':   ('▼  SHORT', C['red']),
            'NEUTRAL': ('◆  NEUTRAL', C['amber']),
            'NO_EDGE': ('── NO EDGE ──', C['text3']),
        }
        dir_text, dir_col = dir_map.get(r.direction, ('?', C['text3']))
        self._dir_var.set(dir_text)
        self._dir_lbl.config(fg=dir_col)

        # Card border color matches direction
        self._card.config(highlightbackground=dir_col)

        # Conviction
        conv = r.conviction
        self._conv_var.set(f'{conv}/10')
        conv_col = (C['green'] if conv >= 7 else C['amber'] if conv >= 4 else C['red'])
        self._conv_lbl.config(fg=conv_col)
        self._conv_bar_frame.update_idletasks()
        bw = self._conv_bar_frame.winfo_width()
        if bw > 1:
            self._conv_bar.place(relwidth=conv / 10.0)
            self._conv_bar.config(bg=conv_col)

        # Metrics
        risk_col = {
            'LOW': C['green'], 'MEDIUM': C['amber'],
            'HIGH': C['red'],  'EXTREME': C['purple'],
        }.get(r.risk_level, C['text3'])
        self._horizon_var.set(r.time_horizon)
        self._risk_var.set(r.risk_level)
        self._size_var.set(f'{r.position_size_pct:.1f}%')
        self._rr_var.set(f'{r.risk_reward:.1f}')

        # Price levels
        from models import DECIMALS
        dec = DECIMALS.get(r.pair, 2)
        self._entry_var.set(f'{r.entry_zone_lo:,.{dec}f} – {r.entry_zone_hi:,.{dec}f}')
        self._sl_var.set(f'{r.stop_loss:,.{dec}f}')
        self._tp1_var.set(f'{r.take_profit_1:,.{dec}f}')
        self._tp2_var.set(f'{r.take_profit_2:,.{dec}f}')

        # Driver
        self._driver_var.set(r.primary_driver or '--')
        self._inval_var.set(r.invalidation or '--')
        self._regime_var.set(f'Regime: {r.regime}')

    def _update_confluence(self, r) -> None:
        C  = self._C
        t  = self._confluence_text
        t.config(state='normal'); t.delete('1.0', 'end')
        if r.confluence:
            for item in r.confluence:
                t.insert('end', f'  • {item}\n')
        else:
            t.insert('end', '  No confluence factors.\n')
        t.config(state='disabled')

    def _update_reasoning(self, r) -> None:
        t = self._reasoning_text
        t.config(state='normal'); t.delete('1.0', 'end')
        chain = r.raw_json.get('reasoning_chain', [])
        if chain:
            for i, step in enumerate(chain, 1):
                t.insert('end', f'[{i}] ', 'step')
                t.insert('end', str(step) + '\n', 'text')
        else:
            t.insert('end', r.primary_driver or 'No reasoning chain available.\n')
        if r.error:
            t.insert('end', f'\n⚠ Parse error: {r.error}\n')
        t.config(state='disabled')

    def _update_history(self) -> None:
        C = self._C
        t = self._history_text
        t.config(state='normal'); t.delete('1.0', 'end')
        for r in self._history:
            tag = r.direction.lower() if r.direction in ('LONG', 'SHORT') else (
                  'neutral' if r.direction == 'NEUTRAL' else 'no_edge')
            dir_sym = {'LONG': '▲', 'SHORT': '▼', 'NEUTRAL': '◆', 'NO_EDGE': '─'}.get(
                r.direction, '?')
            conv_str = f'{r.conviction}/10' if r.conviction > 0 else '--'
            from models import DECIMALS
            dec = DECIMALS.get(r.pair, 2)
            t.insert('end', f'{r.timestamp[11:19]}  ', 'ts')
            t.insert('end', f'{dir_sym} {r.direction:<8}', tag)
            t.insert('end', f'  conv:{conv_str}  ', 'conv')
            t.insert('end', f'  {r.primary_driver[:60]}\n', 'driver')
        t.config(state='disabled')

    def _update_stats(self, gemini) -> None:
        stats = gemini.stats
        chars, tokens = gemini.user_prompt_size()
        self._stats_var.set(
            f'Calls:{stats["calls"]}  Skips:{stats["skips"]}({stats["skip_rate_pct"]:.0f}%)  '
            f'Saved≈{stats["tokens_saved"]}tok  '
            f'Latency:{stats["latency_ms"]:.0f}ms  '
            f'UserTurn≈{tokens}tok  Model:{stats["model"]}'
        )
        self._latency_var.set(f'{stats["latency_ms"]:.0f}ms')
