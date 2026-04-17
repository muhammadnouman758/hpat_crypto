"""
HPAT v6 — views/ai_prediction_view.py  (multi-provider edition)

Supports three AI providers:
  🟦 GEMINI     — Google Gemini (4 models)
  🟠 GROQ       — Groq Cloud ultra-fast inference (5 models)
  🟣 OPENROUTER — 100+ models via OpenRouter (GPT-4o, Claude, Llama, etc.)

UI layout:
  ┌─ Provider Tabs: [GEMINI] [GROQ] [OPENROUTER] ─────────────────────────┐
  │  API KEY ••••••••  👁  ✓from env   [▶ ENABLE]  [⟳ NOW]  ● LIVE      │
  │  MODEL  [Llama 3.3 70B ▼]  Ultra-fast. Best Groq model.              │
  │  ████████████████████████████████████████████████████ warmup bar      │
  ├───────────────────────────────────────────────────────────────────────┤
  │  ▲  LONG          8/10  HORIZON  RISK  SIZE%  R:R                    │
  │  65,400–65,450    SL: 65,150   TP1: 65,800   TP2: 66,200            │
  ├───────────────────────────────────────────────────────────────────────┤
  │  PRIMARY DRIVER                     [timestamp]                       │
  │  CVD bullish divergence with bid wall at 65,200                       │
  ├────────────────────────────┬──────────────────────────────────────────┤
  │  CONFLUENCE FACTORS        │  INVALIDATION  [REGIME]                  │
  │  ✓ RSI oversold bounce     │  Close below 65,100                      │
  │  ✓ OBI bid-heavy +0.42     │                                          │
  ├───────────────────────────────────────────────────────────────────────┤
  │  AI REASONING CHAIN                                                   │
  │  [1] RSI 1m at 28 — oversold, mean reversion probable                │
  │  [2] OBI +0.42 + bid wall at 65,200 — aggressive bid support         │
  ├───────────────────────────────────────────────────────────────────────┤
  │  SIGNAL HISTORY (last 20)  Calls:12 Skips:8(40%) Saved≈1600tok      │
  └───────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Deque, Dict
import collections
import datetime

from views.base_view import BaseView
from models import C, BUS


class AIPredictionView(BaseView):

    MAX_HISTORY = 20
    _SPINNERS   = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()

        self._history:    Deque = collections.deque(maxlen=self.MAX_HISTORY)
        self._enabled     = False
        self._warming_up  = False
        self._warmup_secs = 5
        self._warmup_cnt  = 0
        self._first_call  = True
        self._countdown   = 10
        self._blink_on    = True
        self._spinner_idx = 0

        # Active provider state
        self._active_provider = 'gemini'

        try:
            from ai_engine import _CADENCE as _cad
            self._cadence_map = _cad
        except Exception:
            self._cadence_map = {'VOLATILE': 10, 'TRENDING': 10, 'RANGING': 30}

        self._build_ui(parent)

        self.subscribe('ai_prediction',
                       lambda e: self._schedule(self._on_prediction, e.payload))
        self.subscribe('ai_prediction_error',
                       lambda e: self._schedule(self._on_error, e.payload))
        self._tick()

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self, parent: tk.Widget) -> None:
        C = self._C
        from ai_engine import PROVIDERS

        # ══ SECTION 1: PROVIDER + CONFIG ══════════════════════════════════════
        cfg_outer = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        cfg_outer.pack(fill='x', padx=4, pady=(4, 2))

        # ── Row A: Header + provider tabs ──────────────────────────────────
        row_a = tk.Frame(cfg_outer, bg=C['bg1'], padx=10, pady=(8, 4))
        row_a.pack(fill='x')

        tk.Label(row_a, text='🤖 AI SIGNAL ENGINE', bg=C['bg1'],
                 fg=C['text'], font=(self.MONO, 13, 'bold')).pack(side='left')

        # Provider tab buttons
        self._prov_btns: Dict[str, tk.Button] = {}
        prov_bar = tk.Frame(row_a, bg=C['bg1'])
        prov_bar.pack(side='left', padx=12)

        prov_styles = {
            'gemini':     ('#00eeff', '#001a22', '  GEMINI  '),
            'groq':       ('#ff6b35', '#200e00', '  GROQ  '),
            'openrouter': ('#7c5cfc', '#100a20', '  OPENROUTER  '),
        }
        for prov_id, (fg, bg_act, label) in prov_styles.items():
            b = tk.Button(
                prov_bar, text=label,
                bg=bg_act if prov_id == 'gemini' else C['bg3'],
                fg=fg if prov_id == 'gemini' else C['text3'],
                font=(self.MONO, 9, 'bold'),
                bd=0, padx=6, pady=4, cursor='hand2',
                highlightthickness=1,
                highlightbackground=fg if prov_id == 'gemini' else C['border'],
                command=lambda p=prov_id: self._switch_provider(p),
            )
            b.pack(side='left', padx=(0, 3))
            self._prov_btns[prov_id] = b

        # Active model badge
        self._model_badge_var = tk.StringVar(value='')
        self._model_badge_lbl = tk.Label(row_a, textvariable=self._model_badge_var,
                                          bg=C['bg3'], fg=C['cyan'],
                                          font=(self.MONO, 9, 'bold'), padx=6, pady=2)
        self._model_badge_lbl.pack(side='left', padx=6)

        # Pulse indicator
        self._pulse_var = tk.StringVar(value='')
        tk.Label(row_a, textvariable=self._pulse_var, bg=C['bg1'],
                 fg=C['text3'], font=self.F_TINY).pack(side='right')

        # ── Row B: API Key ─────────────────────────────────────────────────
        row_b = tk.Frame(cfg_outer, bg=C['bg1'], padx=10, pady=(0, 4))
        row_b.pack(fill='x')

        self._key_label = tk.Label(row_b, text='GEMINI KEY', bg=C['bg1'],
                                    fg=C['text3'], font=self.F_LABEL, width=12, anchor='w')
        self._key_label.pack(side='left')

        self._key_var = tk.StringVar(value=os.environ.get('GEMINI_API_KEY', ''))
        self._key_entry = tk.Entry(row_b, textvariable=self._key_var,
                                    show='•', width=44,
                                    bg=C['bg3'], fg=C['text'],
                                    font=self.F_MONO_SM, bd=0,
                                    insertbackground=C['cyan'],
                                    highlightthickness=1,
                                    highlightbackground=C['border2'])
        self._key_entry.pack(side='left', padx=(4, 4))

        self._show_key = False
        tk.Button(row_b, text='👁', bg=C['bg2'], fg=C['text3'],
                  font=self.F_TINY, bd=0, padx=4, cursor='hand2',
                  command=self._toggle_key_vis).pack(side='left', padx=(0, 8))

        self._env_hint_var = tk.StringVar(value='')
        self._env_hint_lbl = tk.Label(row_b, textvariable=self._env_hint_var,
                                       bg=C['bg1'], fg=C['teal'], font=self.F_TINY)
        self._env_hint_lbl.pack(side='left')

        # ── Row C: Model Selector + controls ──────────────────────────────
        row_c = tk.Frame(cfg_outer, bg=C['bg1'], padx=10, pady=(0, 6))
        row_c.pack(fill='x')

        tk.Label(row_c, text='MODEL', bg=C['bg1'], fg=C['text3'],
                 font=self.F_LABEL, width=7, anchor='w').pack(side='left')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('AI.TCombobox', fieldbackground=C['bg3'], background=C['bg3'],
                         foreground=C['text'], selectbackground=C['bg4'],
                         selectforeground=C['cyan'], borderwidth=0, arrowcolor=C['cyan'])
        self._model_var   = tk.StringVar()
        self._model_combo = ttk.Combobox(row_c, textvariable=self._model_var,
                                          state='readonly', width=28,
                                          style='AI.TCombobox', font=self.F_MONO_SM)
        self._model_combo.pack(side='left', padx=(4, 8))
        self._model_combo.bind('<<ComboboxSelected>>', self._on_model_changed)

        self._model_desc_var = tk.StringVar(value='')
        tk.Label(row_c, textvariable=self._model_desc_var, bg=C['bg1'],
                 fg=C['text3'], font=self.F_TINY).pack(side='left', padx=(0, 12))

        # ENABLE / DISABLE
        self._toggle_btn = tk.Button(
            row_c, text='▶  ENABLE',
            bg='#0a0a1a', fg=C['cyan'],
            font=(self.MONO, 10, 'bold'),
            bd=0, padx=14, pady=5, cursor='hand2',
            highlightthickness=1, highlightbackground=C['cyan'],
            command=self._toggle_engine,
        )
        self._toggle_btn.pack(side='left', padx=(0, 5))

        # Force refresh
        self._refresh_btn = tk.Button(
            row_c, text='⟳ NOW', bg=C['bg3'], fg=C['text3'],
            font=(self.MONO, 9, 'bold'), bd=0, padx=8, pady=5,
            cursor='hand2', highlightthickness=1, highlightbackground=C['border'],
            command=self._force_refresh, state='disabled',
        )
        self._refresh_btn.pack(side='left', padx=(0, 10))

        # Status pill
        self._status_pill = tk.Frame(row_c, bg=C['bg3'], padx=10, pady=4,
                                      highlightthickness=1, highlightbackground=C['border'])
        self._status_pill.pack(side='left')
        self._status_dot = tk.Label(self._status_pill, text='●', bg=C['bg3'],
                                     fg=C['text3'], font=self.F_TINY)
        self._status_dot.pack(side='left', padx=(0, 4))
        self._status_var = tk.StringVar(value='OFFLINE')
        self._status_lbl = tk.Label(self._status_pill, textvariable=self._status_var,
                                     bg=C['bg3'], fg=C['text3'],
                                     font=(self.MONO, 10, 'bold'))
        self._status_lbl.pack(side='left')

        # Countdown + latency (right)
        meta_right = tk.Frame(row_c, bg=C['bg1'])
        meta_right.pack(side='right')
        self._latency_var   = tk.StringVar(value='')
        self._countdown_var = tk.StringVar(value='')
        tk.Label(meta_right, textvariable=self._latency_var, bg=C['bg1'],
                 fg=C['text3'], font=self.F_TINY).pack(side='right', padx=4)
        tk.Label(meta_right, textvariable=self._countdown_var, bg=C['bg1'],
                 fg=C['cyan'], font=(self.MONO, 10, 'bold')).pack(side='right', padx=4)

        # Warmup bar
        self._warmup_bar_frame = tk.Frame(cfg_outer, bg=C['bg3'], height=3)
        self._warmup_bar_frame.pack(fill='x')
        self._warmup_bar = tk.Frame(self._warmup_bar_frame, bg=C['cyan'], height=3)
        self._warmup_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # ══ SECTION 2: SIGNAL CARD ════════════════════════════════════════════
        self._card = tk.Frame(parent, bg=C['bg2'],
                               highlightthickness=2, highlightbackground=C['border2'])
        self._card.pack(fill='x', padx=4, pady=(0, 2))

        # Placeholder (shown until first prediction)
        self._placeholder = tk.Frame(self._card, bg=C['bg2'])
        self._placeholder.pack(fill='x', padx=16, pady=20)
        tk.Label(self._placeholder, text='🤖  MULTI-PROVIDER AI SIGNAL ENGINE',
                 bg=C['bg2'], fg=C['text3'],
                 font=(self.MONO, 13, 'bold')).pack()
        tk.Label(self._placeholder,
                 text='Select a provider tab, enter your API key, choose a model,\nthen click ▶ ENABLE to start receiving AI signals.',
                 bg=C['bg2'], fg=C['text3'], font=(self.MONO, 10), justify='center').pack(pady=(6, 0))
        self._warmup_label = tk.Label(self._placeholder, text='',
                                       bg=C['bg2'], fg=C['cyan'],
                                       font=(self.MONO, 11, 'bold'))
        self._warmup_label.pack(pady=(8, 0))

        # Live signal card (hidden until first result)
        self._live_card = tk.Frame(self._card, bg=C['bg2'])

        # Direction + conviction + metrics
        top_row = tk.Frame(self._live_card, bg=C['bg2'])
        top_row.pack(fill='x', padx=12, pady=(10, 4))

        self._dir_var = tk.StringVar(value='── NO SIGNAL ──')
        self._dir_lbl = tk.Label(top_row, textvariable=self._dir_var,
                                  bg=C['bg2'], fg=C['text3'],
                                  font=(self.MONO, 24, 'bold'), width=14, anchor='w')
        self._dir_lbl.pack(side='left')

        # Provider source badge (shown in live card)
        self._source_var = tk.StringVar(value='')
        self._source_lbl = tk.Label(top_row, textvariable=self._source_var,
                                     bg=C['bg3'], fg=C['text3'],
                                     font=(self.MONO, 9, 'bold'), padx=6, pady=2)
        self._source_lbl.pack(side='left', padx=8)

        conv_f = tk.Frame(top_row, bg=C['bg2'])
        conv_f.pack(side='left', padx=8)
        tk.Label(conv_f, text='CONVICTION', bg=C['bg2'],
                 fg=C['text3'], font=self.F_TINY).pack()
        self._conv_var = tk.StringVar(value='--/10')
        self._conv_lbl = tk.Label(conv_f, textvariable=self._conv_var,
                                   bg=C['bg2'], fg=C['text3'],
                                   font=(self.MONO, 20, 'bold'))
        self._conv_lbl.pack()
        self._conv_bar_frame = tk.Frame(conv_f, bg=C['bg3'], height=6)
        self._conv_bar_frame.pack(fill='x', pady=2)
        self._conv_bar_frame.pack_propagate(False)
        self._conv_bar = tk.Frame(self._conv_bar_frame, bg=C['text3'], height=6)
        self._conv_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        for label, vattr, color in [
            ('HORIZON', '_horizon_var', C['text2']),
            ('RISK',    '_risk_var',    C['amber']),
            ('SIZE %',  '_size_var',    C['cyan']),
            ('R : R',   '_rr_var',      C['teal']),
        ]:
            pill = tk.Frame(top_row, bg=C['bg3'], padx=12, pady=6,
                             highlightthickness=1, highlightbackground=C['border'])
            pill.pack(side='left', padx=(0, 4))
            tk.Label(pill, text=label, bg=C['bg3'], fg=C['text3'], font=self.F_TINY).pack()
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(pill, textvariable=v, bg=C['bg3'], fg=color,
                     font=(self.MONO, 13, 'bold')).pack()

        # Price levels
        levels_row = tk.Frame(self._live_card, bg=C['bg2'])
        levels_row.pack(fill='x', padx=12, pady=(0, 10))
        for label, vattr, fg_col, bg_col in [
            ('ENTRY ZONE', '_entry_var', C['text'],  C['bg3']),
            ('STOP LOSS',  '_sl_var',    C['red'],   '#200005'),
            ('TP 1',       '_tp1_var',   C['green'], '#002010'),
            ('TP 2',       '_tp2_var',   C['teal'],  '#001818'),
        ]:
            cell = tk.Frame(levels_row, bg=bg_col, padx=10, pady=6,
                             highlightthickness=1, highlightbackground=fg_col)
            cell.pack(side='left', padx=(0, 4))
            tk.Label(cell, text=label, bg=bg_col, fg=C['text3'], font=self.F_TINY).pack(anchor='w')
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(cell, textvariable=v, bg=bg_col, fg=fg_col,
                     font=(self.MONO, 12, 'bold')).pack(anchor='w')

        # ══ SECTION 3: PRIMARY DRIVER ═════════════════════════════════════════
        drv_outer = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        drv_outer.pack(fill='x', padx=4, pady=(0, 2))
        tk.Frame(drv_outer, bg=C['amber'], width=3).place(x=0, y=0, relheight=1)
        drv_inner = tk.Frame(drv_outer, bg=C['bg1'], padx=12, pady=8)
        drv_inner.pack(fill='x')
        drv_hdr = tk.Frame(drv_inner, bg=C['bg1'])
        drv_hdr.pack(fill='x')
        tk.Label(drv_hdr, text='PRIMARY DRIVER', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._ts_var = tk.StringVar(value='')
        tk.Label(drv_hdr, textvariable=self._ts_var, bg=C['bg1'],
                 fg=C['text3'], font=self.F_TINY).pack(side='right')
        self._driver_var = tk.StringVar(
            value='Select a provider above and click ▶ ENABLE to start receiving signals →')
        self._driver_lbl = tk.Label(drv_inner, textvariable=self._driver_var,
                                     bg=C['bg1'], fg=C['text3'],
                                     font=(self.MONO, 11, 'bold'),
                                     wraplength=1100, justify='left')
        self._driver_lbl.pack(anchor='w', pady=(4, 0))

        # ══ SECTION 4: CONFLUENCE + INVALIDATION ══════════════════════════════
        cf_row = tk.Frame(parent, bg=C['bg'])
        cf_row.pack(fill='x', padx=4, pady=(0, 2))

        cf_left = tk.Frame(cf_row, bg=C['bg1'],
                            highlightthickness=1, highlightbackground=C['border'])
        cf_left.pack(side='left', fill='both', expand=True, padx=(0, 2))
        tk.Frame(cf_left, bg=C['teal'], width=3).place(x=0, y=0, relheight=1)
        cf_l = tk.Frame(cf_left, bg=C['bg1'], padx=12, pady=8)
        cf_l.pack(fill='both', expand=True)
        tk.Label(cf_l, text='CONFLUENCE FACTORS', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(anchor='w')
        self._confluence_text = tk.Text(cf_l, bg=C['bg1'], fg=C['teal'],
                                         font=self.F_MONO_SM, bd=0, height=4,
                                         state='disabled', wrap='word',
                                         selectbackground=C['bg3'])
        self._confluence_text.pack(fill='both', expand=True, pady=(4, 0))

        cf_right = tk.Frame(cf_row, bg=C['bg1'],
                             highlightthickness=1, highlightbackground=C['border'])
        cf_right.pack(side='left', fill='both', expand=True)
        tk.Frame(cf_right, bg=C['red'], width=3).place(x=0, y=0, relheight=1)
        cf_r = tk.Frame(cf_right, bg=C['bg1'], padx=12, pady=8)
        cf_r.pack(fill='both', expand=True)
        inval_hdr = tk.Frame(cf_r, bg=C['bg1'])
        inval_hdr.pack(fill='x')
        tk.Label(inval_hdr, text='INVALIDATION SCENARIO', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._regime_var = tk.StringVar(value='')
        tk.Label(inval_hdr, textvariable=self._regime_var, bg=C['bg1'],
                 fg=C['blue'], font=(self.MONO, 9, 'bold')).pack(side='right')
        self._inval_var = tk.StringVar(value='--')
        tk.Label(cf_r, textvariable=self._inval_var, bg=C['bg1'], fg=C['red'],
                 font=(self.MONO, 11, 'bold'), wraplength=420, justify='left').pack(anchor='w', pady=(4, 0))

        # ══ SECTION 5: REASONING CHAIN ════════════════════════════════════════
        rsn_outer = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        rsn_outer.pack(fill='x', padx=4, pady=(0, 2))
        tk.Frame(rsn_outer, bg=C['blue'], width=3).place(x=0, y=0, relheight=1)
        rsn_inner = tk.Frame(rsn_outer, bg=C['bg1'], padx=12, pady=8)
        rsn_inner.pack(fill='x')
        tk.Label(rsn_inner, text='AI REASONING CHAIN', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(anchor='w')
        rsb = tk.Scrollbar(rsn_inner, bg=C['bg2'], troughcolor=C['bg3'])
        rsb.pack(side='right', fill='y')
        self._reasoning_text = tk.Text(rsn_inner, bg=C['bg1'], fg=C['text2'],
                                        font=self.F_MONO_SM, bd=0, height=4,
                                        yscrollcommand=rsb.set, state='disabled',
                                        wrap='word', selectbackground=C['bg3'])
        self._reasoning_text.pack(fill='x', pady=(4, 0))
        rsb.config(command=self._reasoning_text.yview)
        for tag, col in [('step_num', C['purple']), ('step_txt', C['text2']), ('err', C['red'])]:
            self._reasoning_text.tag_configure(tag, foreground=col)

        # ══ SECTION 6: SIGNAL HISTORY ═════════════════════════════════════════
        hist_outer = tk.Frame(parent, bg=C['bg1'],
                               highlightthickness=1, highlightbackground=C['border'])
        hist_outer.pack(fill='both', expand=True, padx=4, pady=(0, 2))
        tk.Frame(hist_outer, bg=C['text3'], width=3).place(x=0, y=0, relheight=1)
        hist_inner = tk.Frame(hist_outer, bg=C['bg1'], padx=12, pady=8)
        hist_inner.pack(fill='both', expand=True)
        hist_hdr = tk.Frame(hist_inner, bg=C['bg1'])
        hist_hdr.pack(fill='x')
        tk.Label(hist_hdr, text=f'SIGNAL HISTORY  (last {self.MAX_HISTORY})',
                 bg=C['bg1'], fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._stats_var = tk.StringVar(value='')
        tk.Label(hist_hdr, textvariable=self._stats_var, bg=C['bg1'],
                 fg=C['text3'], font=self.F_TINY).pack(side='right')
        hsb = tk.Scrollbar(hist_inner, bg=C['bg2'], troughcolor=C['bg3'])
        hsb.pack(side='right', fill='y')
        self._history_text = tk.Text(hist_inner, bg=C['bg1'], fg=C['text'],
                                      font=self.F_MONO_SM, bd=0, height=7,
                                      yscrollcommand=hsb.set, state='disabled',
                                      wrap='none', selectbackground=C['bg3'])
        self._history_text.pack(fill='both', expand=True, pady=(4, 0))
        hsb.config(command=self._history_text.yview)
        for tag, col in [
            ('long',    C['green']), ('short',  C['red']),
            ('neutral', C['amber']), ('no_edge',C['text3']),
            ('ts',      C['text3']), ('conv',   C['purple']),
            ('driver',  C['text2']), ('rr',     C['teal']),
            ('prov_g',  '#00eeff'), ('prov_q',  '#ff6b35'),
            ('prov_o',  '#7c5cfc'), ('regime',  C['blue']),
        ]:
            self._history_text.tag_configure(tag, foreground=col)

        # Initialise provider UI to gemini
        self._switch_provider('gemini', silent=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PROVIDER SWITCHING
    # ─────────────────────────────────────────────────────────────────────────

    def _switch_provider(self, prov_id: str, silent: bool = False) -> None:
        from ai_engine import PROVIDERS
        C    = self._C
        prov = PROVIDERS[prov_id]
        self._active_provider = prov_id

        # Update tab button appearances
        styles = {
            'gemini':     ('#00eeff', '#001a22'),
            'groq':       ('#ff6b35', '#200e00'),
            'openrouter': ('#7c5cfc', '#100a20'),
        }
        for pid, btn in self._prov_btns.items():
            fg, bg_act = styles[pid]
            if pid == prov_id:
                btn.config(bg=bg_act, fg=fg,
                           highlightbackground=fg)
            else:
                btn.config(bg=C['bg3'], fg=C['text3'],
                           highlightbackground=C['border'])

        # Update key label + env var hint
        env_key  = prov.get('env_key', '')
        env_val  = os.environ.get(env_key, '')
        self._key_label.config(text=f'{prov_id.upper()} KEY')
        if env_val:
            self._key_var.set(env_val)
            self._env_hint_var.set(f'✓ from env ({env_key})')
            self._env_hint_lbl.config(fg=C['teal'])
        else:
            self._env_hint_var.set(f'or set {env_key} env var')
            self._env_hint_lbl.config(fg=C['text3'])

        # Update warmup bar and toggle colour
        prov_color = prov.get('color', C['cyan'])
        self._warmup_bar.config(bg=prov_color)
        self._toggle_btn.config(fg=prov_color,
                                 highlightbackground=prov_color)

        # Populate model dropdown
        models      = prov.get('models', [])
        model_labels = [m['label'] for m in models]
        self._model_combo.config(values=model_labels)
        if model_labels:
            self._model_var.set(model_labels[0])
            self._refresh_model_badge()

        # If engine is live and provider changed, stop + require re-enable
        if not silent and self._enabled:
            from ai_engine import AI
            AI.stop()
            self._enabled = False
            self._toggle_btn.config(text='▶  ENABLE')
            self._refresh_btn.config(state='disabled', fg=C['text3'])
            self._set_status('OFFLINE', C['text3'], pill_bg=C['bg3'])
            self._countdown_var.set('')

    def _refresh_model_badge(self) -> None:
        from ai_engine import PROVIDERS
        C    = self._C
        prov = PROVIDERS.get(self._active_provider, {})
        models = prov.get('models', [])
        label  = self._model_var.get()
        model  = next((m for m in models if m['label'] == label), models[0] if models else {})
        badge  = model.get('badge', '')
        tier   = model.get('tier', '')
        color  = prov.get('color', C['cyan'])
        self._model_badge_var.set(f' {badge}  {tier} ')
        self._model_badge_lbl.config(fg=color)
        self._model_desc_var.set(model.get('desc', ''))

    def _get_selected_model_id(self) -> str:
        from ai_engine import PROVIDERS
        prov   = PROVIDERS.get(self._active_provider, {})
        models = prov.get('models', [])
        label  = self._model_var.get()
        m      = next((m for m in models if m['label'] == label), models[0] if models else {})
        return m.get('id', '')

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE CONTROLS
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_key_vis(self) -> None:
        self._show_key = not self._show_key
        self._key_entry.config(show='' if self._show_key else '•')

    def _toggle_engine(self) -> None:
        from ai_engine import AI
        if self._enabled:
            self._disable_engine(AI)
        else:
            self._do_configure()

    def _disable_engine(self, ai) -> None:
        C = self._C
        ai.stop()
        self._enabled    = False
        self._warming_up = False
        self._toggle_btn.config(text='▶  ENABLE', bg='#0a0a1a')
        self._refresh_btn.config(state='disabled', fg=C['text3'])
        self._set_status('OFFLINE', C['text3'], pill_bg=C['bg3'])
        self._countdown_var.set('')
        self._pulse_var.set('')
        self._warmup_bar.place(relwidth=0.0)

    def _do_configure(self) -> None:
        from ai_engine import AI, PROVIDERS
        C         = self._C
        key       = self._key_var.get().strip()
        prov_id   = self._active_provider
        model_id  = self._get_selected_model_id()
        prov_col  = PROVIDERS[prov_id].get('color', C['cyan'])

        if not key:
            self._key_entry.config(highlightbackground=C['red'])
            self._set_status('⚠  NO API KEY', C['red'], pill_bg='#200000')
            return
        self._key_entry.config(highlightbackground=C['border2'])
        if not model_id:
            self._set_status('⚠  NO MODEL SELECTED', C['red'], pill_bg='#200000')
            return

        # Show connecting state
        self._toggle_btn.config(text='⟳  CONNECTING', fg=C['amber'], state='disabled')
        self._set_status('CONNECTING…', C['amber'], pill_bg='#1a1000')
        self._app.update_idletasks()

        ok, err_msg = AI.configure(prov_id, key, model_id)
        self._toggle_btn.config(state='normal')

        if ok:
            self._enabled    = True
            self._warming_up = True
            self._warmup_cnt = 0
            self._first_call = True
            self._countdown  = self._warmup_secs
            self._toggle_btn.config(text='■  DISABLE', bg='#200005', fg=C['red'])
            self._refresh_btn.config(state='normal', fg=prov_col)
            self._set_status('WARMING UP', C['amber'], pill_bg='#1a1000')
            prov_name = PROVIDERS[prov_id]['name']
            self._warmup_label.config(
                text=f'⟳  Connecting to {prov_name}… first signal in ~5s',
                fg=prov_col)
            # ── CRITICAL: schedule run() on the live feed loop ────────────────
            if hasattr(self._app, '_feed'):
                self._app._feed.start_ai()
        else:
            self._toggle_btn.config(text='▶  ENABLE', bg='#0a0a1a', fg=prov_col)
            short_err = (err_msg[:60] + '…') if len(err_msg) > 60 else err_msg
            self._set_status(f'⚠  {short_err}', C['red'], pill_bg='#200000')
            self._driver_var.set(f'⚠  Config error: {err_msg}')
            self._driver_lbl.config(fg=C['red'])

    def _on_model_changed(self, event=None) -> None:
        self._refresh_model_badge()
        if self._enabled:
            from ai_engine import AI
            AI.stop()
            AI.model_id = self._get_selected_model_id()
            if AI._client:
                AI._client.model_id = AI.model_id
                AI.enabled = True
            if hasattr(self._app, '_feed'):
                self._app._feed.start_ai()

    def _force_refresh(self) -> None:
        try:
            from ai_engine import AI
            AI._last_feat = {}
            task = AI._task
            if task is None or task.done():
                if hasattr(self._app, '_feed'):
                    self._app._feed.start_ai()
            prov = self._active_provider
            from ai_engine import PROVIDERS
            col = PROVIDERS.get(prov, {}).get('color', self._C['cyan'])
            self._set_status('FORCED ⟳', col, pill_bg=self._C['bg3'])
            self._app.after(1500, lambda: self._set_status(
                'LIVE', col, pill_bg='#060010'))
        except Exception as e:
            self._set_status(f'⚠ {str(e)[:30]}', self._C['red'])

    def _set_status(self, text: str, color: str, pill_bg: str = None) -> None:
        C  = self._C
        bg = pill_bg or C['bg3']
        self._status_var.set(text)
        self._status_lbl.config(fg=color, bg=bg)
        self._status_dot.config(fg=color, bg=bg)
        self._status_pill.config(bg=bg, highlightbackground=color)

    # ─────────────────────────────────────────────────────────────────────────
    # TICK / ANIMATION
    # ─────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        C = self._C
        if self._enabled:
            if self._warming_up:
                self._warmup_cnt += 1
                prog = min(1.0, self._warmup_cnt / (self._warmup_secs * 10))
                self._warmup_bar.place(relwidth=prog)
                spin = self._SPINNERS[self._warmup_cnt % len(self._SPINNERS)]
                rem  = max(0, self._warmup_secs - self._warmup_cnt // 10)
                self._warmup_label.config(text=f'{spin}  Warming up… {rem}s')
                if prog >= 1.0:
                    self._warmup_bar.place(relwidth=0.0)

            self._countdown = max(0, self._countdown - 1)
            self._countdown_var.set(f'NEXT  {self._countdown}s' if self._countdown > 0 else 'ANALYSING…')

            self._blink_on = not self._blink_on
            if not self._warming_up:
                from ai_engine import PROVIDERS
                col = PROVIDERS.get(self._active_provider, {}).get('color', C['cyan'])
                self._status_dot.config(fg=col if self._blink_on else C['bg3'])

            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNERS)
            self._pulse_var.set(f'{self._SPINNERS[self._spinner_idx]}  10s PULSE')

        self._app.after(100, self._tick)

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _on_prediction(self, result) -> None:
        from ai_engine import AI, PROVIDERS
        C     = self._C
        prov  = result.provider if hasattr(result, 'provider') else self._active_provider
        prov_col = PROVIDERS.get(prov, {}).get('color', C['cyan'])

        if self._first_call and not result.skipped:
            self._first_call  = False
            self._warming_up  = False
            self._warmup_label.config(text='')
            self._warmup_bar.place(relwidth=0.0)
            self._placeholder.pack_forget()
            self._live_card.pack(fill='x')
            self._driver_lbl.config(fg=C['text2'])
            self._set_status('LIVE', prov_col, pill_bg='#060010')

        if not result.skipped:
            self._history.appendleft(result)
            self._update_card(result, prov_col)
            self._update_confluence(result)
            self._update_reasoning(result)
            self._update_history()
        else:
            if not self._first_call:
                self._set_status(f'CACHED  ({result.skip_reason[:18]})',
                                 C['text3'], pill_bg=C['bg3'])
                self._app.after(2000, lambda: self._set_status(
                    'LIVE', prov_col, pill_bg='#060010'))

        self._update_stats(AI)
        regime  = getattr(result, 'regime', 'RANGING')
        self._countdown = self._cadence_map.get(regime, 30)

    def _on_error(self, msg: str) -> None:
        C = self._C
        self._set_status(f'⚠  {str(msg)[:35]}', C['red'], pill_bg='#200000')
        self._driver_var.set(f'⚠  API error: {str(msg)[:80]}')
        self._driver_lbl.config(fg=C['red'])

    # ─────────────────────────────────────────────────────────────────────────
    # CONTENT UPDATES
    # ─────────────────────────────────────────────────────────────────────────

    def _update_card(self, r, prov_col: str) -> None:
        C = self._C
        dir_styles = {
            'LONG':    ('▲  LONG',    C['green'], '#001f0a', C['green']),
            'SHORT':   ('▼  SHORT',   C['red'],   '#1a0005', C['red']),
            'NEUTRAL': ('◆  NEUTRAL', C['amber'], '#1a1000', C['amber']),
            'NO_EDGE': ('─  NO EDGE', C['text3'], C['bg2'],  C['border2']),
        }
        dir_text, dir_fg, card_bg, card_border = dir_styles.get(
            r.direction, ('?', C['text3'], C['bg2'], C['border2']))

        self._dir_var.set(dir_text)
        self._dir_lbl.config(fg=dir_fg)
        self._card.config(bg=card_bg, highlightbackground=card_border)
        self._live_card.config(bg=card_bg)

        # Source badge shows provider + model badge
        from ai_engine import ALL_MODELS, PROVIDERS
        model_info = ALL_MODELS.get(getattr(r, 'model_id', ''), {})
        badge = model_info.get('badge', getattr(r, 'model_id', '')[:10])
        self._source_var.set(f' {badge} ')
        self._source_lbl.config(fg=prov_col, bg=C['bg3'])

        conv = r.conviction
        self._conv_var.set(f'{conv}/10')
        conv_col = C['green'] if conv >= 7 else C['amber'] if conv >= 4 else C['red']
        self._conv_lbl.config(fg=conv_col)
        self._conv_bar_frame.update_idletasks()
        bw = self._conv_bar_frame.winfo_width()
        if bw > 1:
            self._conv_bar.place(relwidth=conv / 10.0)
            self._conv_bar.config(bg=conv_col)

        self._horizon_var.set(r.time_horizon)
        self._risk_var.set(r.risk_level)
        self._size_var.set(f'{r.position_size_pct:.1f}%')
        self._rr_var.set(f'{r.risk_reward:.1f}×')

        from models import DECIMALS
        dec = DECIMALS.get(r.pair, 2)
        self._entry_var.set(f'{r.entry_zone_lo:,.{dec}f} – {r.entry_zone_hi:,.{dec}f}')
        self._sl_var.set(f'{r.stop_loss:,.{dec}f}')
        self._tp1_var.set(f'{r.take_profit_1:,.{dec}f}')
        self._tp2_var.set(f'{r.take_profit_2:,.{dec}f}')

        self._driver_var.set(r.primary_driver or '--')
        self._driver_lbl.config(fg=self._C['text'])
        self._ts_var.set(f'  {r.timestamp[11:19]} UTC')
        self._inval_var.set(r.invalidation or '--')
        self._regime_var.set(r.regime or '')

    def _update_confluence(self, r) -> None:
        C = self._C
        t = self._confluence_text
        t.config(state='normal'); t.delete('1.0', 'end')
        t.tag_configure('ok', foreground=C['teal'])
        if r.confluence:
            for item in r.confluence:
                t.insert('end', '  ✓  ', 'ok')
                t.insert('end', item + '\n')
        else:
            t.insert('end', '  No confluence factors reported.\n')
        t.config(state='disabled')

    def _update_reasoning(self, r) -> None:
        t = self._reasoning_text
        t.config(state='normal'); t.delete('1.0', 'end')
        why   = r.reasoning or ''
        steps = [s.strip() for s in why.split('|') if s.strip()] if why else []
        if steps:
            for i, step in enumerate(steps, 1):
                t.insert('end', f'[{i}]  ', 'step_num')
                t.insert('end', step + '\n', 'step_txt')
        elif r.primary_driver:
            t.insert('end', r.primary_driver, 'step_txt')
        else:
            t.insert('end', 'No reasoning available.', 'step_txt')
        if r.error:
            t.insert('end', f'\n⚠  {r.error}\n', 'err')
        t.config(state='disabled')

    def _update_history(self) -> None:
        C = self._C
        t = self._history_text
        t.config(state='normal'); t.delete('1.0', 'end')
        if not self._history:
            t.insert('end', '  No predictions yet.\n', 'ts')
            t.config(state='disabled')
            return
        from models import DECIMALS
        from ai_engine import PROVIDERS
        for r in self._history:
            prov    = getattr(r, 'provider', 'gemini')
            prov_tag = f'prov_{prov[0]}'   # prov_g, prov_q, prov_o
            dir_tag = r.direction.lower() if r.direction in ('LONG','SHORT','NEUTRAL') else 'no_edge'
            sym     = {'LONG':'▲','SHORT':'▼','NEUTRAL':'◆','NO_EDGE':'─'}.get(r.direction,'?')
            rr      = f'{r.risk_reward:.1f}×' if r.risk_reward else '--'
            t.insert('end', f'  {r.timestamp[11:19]}  ', 'ts')
            t.insert('end', f'{sym} {r.direction:<8}', dir_tag)
            t.insert('end', f'  {r.conviction}/10  ', 'conv')
            t.insert('end', f'RR:{rr}  ', 'rr')
            t.insert('end', f'{prov.upper():<12}', prov_tag)
            t.insert('end', f'{r.regime:<10}  ', 'regime')
            t.insert('end', f'{r.primary_driver[:45]}\n', 'driver')
        t.config(state='disabled')

    def _update_stats(self, ai) -> None:
        stats  = ai.stats
        _, tok = ai.user_prompt_size()
        pname  = stats.get('provider_name', stats.get('provider', ''))
        badge  = stats.get('model_badge', '')
        self._stats_var.set(
            f'{pname} {badge}  │  '
            f'Calls:{stats["calls"]}  '
            f'Skips:{stats["skips"]}({stats["skip_rate_pct"]:.0f}%)  '
            f'Saved≈{stats["tokens_saved"]}tok  '
            f'{stats["latency_ms"]:.0f}ms  '
            f'~{tok}tok/call'
        )
        if stats['latency_ms'] > 0:
            self._latency_var.set(f'{stats["latency_ms"]:.0f}ms')
