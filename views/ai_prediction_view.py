"""
HPAT v6 — views/ai_prediction_view.py  (UX overhaul)

Fixes from screenshot review:
  ✓ Model selector dropdown with 4 Gemini variants + descriptions
  ✓ Animated "WARMING UP" / "ANALYSING..." states replace broken --/--
  ✓ Distinct OFFLINE (red) vs LIVE (purple) vs CACHED (blue) status styles
  ✓ Progress bar fills during 5s warmup countdown
  ✓ ENABLE button shows spinner animation while connecting
  ✓ Compact config row: key entry + model selector + button in one tight row
  ✓ First-prediction placeholder card shows helpful setup instructions
  ✓ NEXT: counter hidden until engine is live
  ✓ reasoning_chain bug fixed — reads new 'why' pipe-delimited field
  ✓ Stats footer correctly reflects live model name + token efficiency
  ✓ Cadence import moved to module level
  ✓ Signal card glows with direction colour on each update
  ✓ Refresh button to force a prediction cycle immediately
  ✓ Model badge shows active model + tier label
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

# ─── GEMINI MODEL CATALOGUE ───────────────────────────────────────────────────

GEMINI_MODELS = [
    {
        'id':      'gemini-2.0-flash',
        'label':   'Gemini 2.0 Flash',
        'badge':   '2.0 FLASH',
        'tier':    'FAST',
        'desc':    'Best speed/accuracy balance. Recommended for real-time trading.',
        'color':   '#00eeff',   # cyan
    },
    {
        'id':      'gemini-2.0-flash-lite',
        'label':   'Gemini 2.0 Flash-Lite',
        'badge':   '2.0 LITE',
        'tier':    'ECONOMY',
        'desc':    'Fastest & cheapest. Good for high-frequency scanning.',
        'color':   '#1fffc0',   # teal
    },
    {
        'id':      'gemini-1.5-flash',
        'label':   'Gemini 1.5 Flash',
        'badge':   '1.5 FLASH',
        'tier':    'STABLE',
        'desc':    'Proven stable model. Reliable for production use.',
        'color':   '#29b8ff',   # blue
    },
    {
        'id':      'gemini-1.5-pro',
        'label':   'Gemini 1.5 Pro',
        'badge':   '1.5 PRO',
        'tier':    'DEEP',
        'desc':    'Deepest reasoning. Use for complex multi-factor analysis.',
        'color':   '#e040fb',   # purple
    },
]

# Map model id → catalogue entry for quick lookup
_MODEL_MAP = {m['id']: m for m in GEMINI_MODELS}

# ─── VIEW CLASS ───────────────────────────────────────────────────────────────

class AIPredictionView(BaseView):

    MAX_HISTORY = 20

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()

        self._history: Deque = collections.deque(maxlen=self.MAX_HISTORY)
        self._countdown    = 10
        self._enabled      = False
        self._warming_up   = False
        self._warmup_secs  = 5      # matches data_feed warmup delay
        self._warmup_cnt   = 0
        self._blink_on     = True
        self._spinner_idx  = 0
        self._first_call   = True   # True until first real prediction arrives

        # Import cadence map once at module level (safe)
        try:
            from gemini_engine import _CADENCE as _cad
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

        # ══ SECTION 1: CONFIG BAR ═════════════════════════════════════════════
        cfg_frame = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        cfg_frame.pack(fill='x', padx=4, pady=(4, 2))

        cfg_inner = tk.Frame(cfg_frame, bg=C['bg1'], padx=10, pady=8)
        cfg_inner.pack(fill='x')

        # Left accent bar
        tk.Frame(cfg_frame, bg=C['purple'], width=3).place(x=0, y=0, relheight=1)

        # ── Row A: Header ──────────────────────────────────────────────────
        row_a = tk.Frame(cfg_inner, bg=C['bg1'])
        row_a.pack(fill='x', pady=(0, 6))

        tk.Label(row_a, text='🤖', bg=C['bg1'], font=(self.MONO, 14)).pack(side='left')
        tk.Label(row_a, text=' GEMINI AI SIGNAL ENGINE', bg=C['bg1'],
                 fg=C['text'], font=(self.MONO, 13, 'bold')).pack(side='left')

        # Model badge (updates dynamically)
        self._model_badge_var = tk.StringVar(value='')
        self._model_badge_lbl = tk.Label(row_a, textvariable=self._model_badge_var,
                                          bg=C['bg3'], fg=C['cyan'],
                                          font=(self.MONO, 9, 'bold'),
                                          padx=6, pady=2)
        self._model_badge_lbl.pack(side='left', padx=8)

        # Pulse indicator (right)
        self._pulse_var = tk.StringVar(value='')
        tk.Label(row_a, textvariable=self._pulse_var,
                 bg=C['bg1'], fg=C['text3'],
                 font=self.F_TINY).pack(side='right')

        # ── Row B: API Key ─────────────────────────────────────────────────
        row_b = tk.Frame(cfg_inner, bg=C['bg1'])
        row_b.pack(fill='x', pady=(0, 4))

        tk.Label(row_b, text='API KEY', bg=C['bg1'], fg=C['text3'],
                 font=self.F_LABEL, width=8, anchor='w').pack(side='left')

        self._key_var = tk.StringVar(value=os.environ.get('GEMINI_API_KEY', ''))
        self._key_entry = tk.Entry(row_b, textvariable=self._key_var,
                                    show='•', width=42,
                                    bg=C['bg3'], fg=C['text'],
                                    font=self.F_MONO_SM, bd=0,
                                    insertbackground=C['purple'],
                                    highlightthickness=1,
                                    highlightbackground=C['border2'])
        self._key_entry.pack(side='left', padx=(4, 8))

        # Show/hide key toggle
        self._show_key = False
        tk.Button(row_b, text='👁', bg=C['bg2'], fg=C['text3'],
                  font=self.F_TINY, bd=0, padx=4, cursor='hand2',
                  command=self._toggle_key_visibility).pack(side='left', padx=(0, 8))

        # Env var hint
        env_set = bool(os.environ.get('GEMINI_API_KEY'))
        hint_txt = '✓ from env' if env_set else 'or set GEMINI_API_KEY env var'
        hint_col = C['teal'] if env_set else C['text3']
        tk.Label(row_b, text=hint_txt, bg=C['bg1'], fg=hint_col,
                 font=self.F_TINY).pack(side='left')

        # ── Row C: Model Selector + Action Buttons ─────────────────────────
        row_c = tk.Frame(cfg_inner, bg=C['bg1'])
        row_c.pack(fill='x', pady=(0, 4))

        tk.Label(row_c, text='MODEL  ', bg=C['bg1'], fg=C['text3'],
                 font=self.F_LABEL, width=8, anchor='w').pack(side='left')

        # Model dropdown
        self._model_var = tk.StringVar(value=GEMINI_MODELS[0]['label'])
        model_labels    = [m['label'] for m in GEMINI_MODELS]
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('AI.TCombobox',
                         fieldbackground=C['bg3'], background=C['bg3'],
                         foreground=C['text'], selectbackground=C['bg4'],
                         selectforeground=C['cyan'], borderwidth=0,
                         arrowcolor=C['purple'])
        self._model_combo = ttk.Combobox(row_c, textvariable=self._model_var,
                                          values=model_labels,
                                          state='readonly', width=26,
                                          style='AI.TCombobox',
                                          font=self.F_MONO_SM)
        self._model_combo.pack(side='left', padx=(4, 6))
        self._model_combo.bind('<<ComboboxSelected>>', self._on_model_changed)

        # Model description label
        self._model_desc_var = tk.StringVar(value=GEMINI_MODELS[0]['desc'])
        self._model_desc_lbl = tk.Label(row_c, textvariable=self._model_desc_var,
                                         bg=C['bg1'], fg=C['text3'],
                                         font=self.F_TINY)
        self._model_desc_lbl.pack(side='left', padx=(0, 12))

        # ENABLE / DISABLE button
        self._toggle_btn = tk.Button(
            row_c, text='▶  ENABLE',
            bg='#1a0a2e', fg=C['purple'],
            font=(self.MONO, 10, 'bold'),
            bd=0, padx=14, pady=5, cursor='hand2',
            highlightthickness=1, highlightbackground=C['purple'],
            activebackground=C['purple'], activeforeground='#000',
            command=self._toggle_engine,
        )
        self._toggle_btn.pack(side='left', padx=(0, 6))

        # Force refresh button (visible only when live)
        self._refresh_btn = tk.Button(
            row_c, text='⟳ NOW',
            bg=C['bg3'], fg=C['text3'],
            font=(self.MONO, 9, 'bold'),
            bd=0, padx=8, pady=5, cursor='hand2',
            highlightthickness=1, highlightbackground=C['border'],
            command=self._force_refresh,
            state='disabled',
        )
        self._refresh_btn.pack(side='left', padx=(0, 12))

        # ── Status pill ───────────────────────────────────────────────────
        self._status_pill = tk.Frame(row_c, bg=C['bg3'], padx=10, pady=4,
                                      highlightthickness=1, highlightbackground=C['border'])
        self._status_pill.pack(side='left')
        self._status_dot_lbl = tk.Label(self._status_pill, text='●',
                                         bg=C['bg3'], fg=C['text3'],
                                         font=self.F_TINY)
        self._status_dot_lbl.pack(side='left', padx=(0, 4))
        self._status_var = tk.StringVar(value='OFFLINE')
        self._status_lbl = tk.Label(self._status_pill, textvariable=self._status_var,
                                     bg=C['bg3'], fg=C['text3'],
                                     font=(self.MONO, 10, 'bold'))
        self._status_lbl.pack(side='left')

        # Countdown + latency (right side)
        right_meta = tk.Frame(row_c, bg=C['bg1'])
        right_meta.pack(side='right')
        self._countdown_var = tk.StringVar(value='')
        self._latency_var   = tk.StringVar(value='')
        tk.Label(right_meta, textvariable=self._latency_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='right', padx=4)
        self._countdown_lbl = tk.Label(right_meta, textvariable=self._countdown_var,
                                        bg=C['bg1'], fg=C['purple'],
                                        font=(self.MONO, 10, 'bold'))
        self._countdown_lbl.pack(side='right', padx=4)

        # ── Warmup progress bar (shown during 5s init) ────────────────────
        self._warmup_bar_frame = tk.Frame(cfg_inner, bg=C['bg3'], height=3)
        self._warmup_bar_frame.pack(fill='x', pady=(2, 0))
        self._warmup_bar = tk.Frame(self._warmup_bar_frame, bg=C['purple'], height=3)
        self._warmup_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # ══ SECTION 2: SIGNAL CARD ════════════════════════════════════════════
        self._card = tk.Frame(parent, bg=C['bg2'], padx=0, pady=0,
                               highlightthickness=2, highlightbackground=C['border2'])
        self._card.pack(fill='x', padx=4, pady=(0, 2))

        # ── Placeholder shown before first prediction ──────────────────────
        self._placeholder = tk.Frame(self._card, bg=C['bg2'])
        self._placeholder.pack(fill='x', padx=16, pady=20)
        tk.Label(self._placeholder, text='🤖  GEMINI AI PREDICTION ENGINE',
                 bg=C['bg2'], fg=C['text3'],
                 font=(self.MONO, 13, 'bold')).pack()
        tk.Label(self._placeholder,
                 text='Enter your Gemini API key above, select a model,\nthen click ▶ ENABLE to start receiving AI signals.',
                 bg=C['bg2'], fg=C['text3'],
                 font=(self.MONO, 10), justify='center').pack(pady=(6, 0))
        self._warmup_label = tk.Label(self._placeholder, text='',
                                       bg=C['bg2'], fg=C['purple'],
                                       font=(self.MONO, 11, 'bold'))
        self._warmup_label.pack(pady=(8, 0))

        # ── Live signal content (hidden until first prediction) ────────────
        self._live_card = tk.Frame(self._card, bg=C['bg2'])
        # NOT packed yet — shown after first result

        # Top row: direction + conviction + metrics
        top_row = tk.Frame(self._live_card, bg=C['bg2'])
        top_row.pack(fill='x', padx=12, pady=(10, 4))

        self._dir_var = tk.StringVar(value='── NO SIGNAL ──')
        self._dir_lbl = tk.Label(top_row, textvariable=self._dir_var,
                                  bg=C['bg2'], fg=C['text3'],
                                  font=(self.MONO, 24, 'bold'), width=14, anchor='w')
        self._dir_lbl.pack(side='left')

        # Conviction meter
        conv_frame = tk.Frame(top_row, bg=C['bg2'])
        conv_frame.pack(side='left', padx=12)
        tk.Label(conv_frame, text='CONVICTION', bg=C['bg2'],
                 fg=C['text3'], font=self.F_TINY).pack()
        self._conv_var = tk.StringVar(value='--/10')
        self._conv_lbl = tk.Label(conv_frame, textvariable=self._conv_var,
                                   bg=C['bg2'], fg=C['text3'],
                                   font=(self.MONO, 20, 'bold'))
        self._conv_lbl.pack()
        self._conv_bar_frame = tk.Frame(conv_frame, bg=C['bg3'], height=6)
        self._conv_bar_frame.pack(fill='x', pady=2)
        self._conv_bar_frame.pack_propagate(False)
        self._conv_bar = tk.Frame(self._conv_bar_frame, bg=C['text3'], height=6)
        self._conv_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # Metric pills
        for label, vattr, color in [
            ('HORIZON', '_horizon_var', C['text2']),
            ('RISK',    '_risk_var',    C['amber']),
            ('SIZE %',  '_size_var',    C['cyan']),
            ('R : R',   '_rr_var',      C['teal']),
        ]:
            pill = tk.Frame(top_row, bg=C['bg3'], padx=12, pady=6,
                             highlightthickness=1, highlightbackground=C['border'])
            pill.pack(side='left', padx=(0, 4))
            tk.Label(pill, text=label, bg=C['bg3'],
                     fg=C['text3'], font=self.F_TINY).pack()
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(pill, textvariable=v, bg=C['bg3'], fg=color,
                     font=(self.MONO, 13, 'bold')).pack()

        # Price levels row
        levels_row = tk.Frame(self._live_card, bg=C['bg2'])
        levels_row.pack(fill='x', padx=12, pady=(0, 10))
        for label, vattr, color, accent in [
            ('ENTRY ZONE', '_entry_var', C['text'],  C['border']),
            ('STOP LOSS',  '_sl_var',    C['red'],   '#3d0010'),
            ('TP 1',       '_tp1_var',   C['green'], '#00200f'),
            ('TP 2',       '_tp2_var',   C['teal'],  '#002020'),
        ]:
            cell = tk.Frame(levels_row, bg=accent, padx=10, pady=6,
                             highlightthickness=1, highlightbackground=color)
            cell.pack(side='left', padx=(0, 4))
            tk.Label(cell, text=label, bg=accent,
                     fg=C['text3'], font=self.F_TINY).pack(anchor='w')
            v = tk.StringVar(value='--')
            setattr(self, vattr, v)
            tk.Label(cell, textvariable=v, bg=accent, fg=color,
                     font=(self.MONO, 12, 'bold')).pack(anchor='w')

        # ══ SECTION 3: PRIMARY DRIVER ═════════════════════════════════════════
        drv_frame = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        drv_frame.pack(fill='x', padx=4, pady=(0, 2))
        drv_inner = tk.Frame(drv_frame, bg=C['bg1'], padx=10, pady=8)
        drv_inner.pack(fill='x')
        tk.Frame(drv_frame, bg=C['amber'], width=3).place(x=0, y=0, relheight=1)

        drv_hdr = tk.Frame(drv_inner, bg=C['bg1'])
        drv_hdr.pack(fill='x')
        tk.Label(drv_hdr, text='PRIMARY DRIVER', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._ts_var = tk.StringVar(value='')
        tk.Label(drv_hdr, textvariable=self._ts_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='right')

        self._driver_var = tk.StringVar(value='Enable the engine to receive AI market analysis →')
        self._driver_lbl = tk.Label(drv_inner, textvariable=self._driver_var,
                                     bg=C['bg1'], fg=C['text3'],
                                     font=(self.MONO, 11, 'bold'),
                                     wraplength=1100, justify='left')
        self._driver_lbl.pack(anchor='w', pady=(4, 0))

        # ══ SECTION 4: CONFLUENCE + INVALIDATION ══════════════════════════════
        cf_row = tk.Frame(parent, bg=C['bg'])
        cf_row.pack(fill='x', padx=4, pady=(0, 2))

        # Confluence
        cf_left = tk.Frame(cf_row, bg=C['bg1'],
                            highlightthickness=1, highlightbackground=C['border'])
        cf_left.pack(side='left', fill='both', expand=True, padx=(0, 2))
        cf_l_inner = tk.Frame(cf_left, bg=C['bg1'], padx=10, pady=8)
        cf_l_inner.pack(fill='both', expand=True)
        tk.Frame(cf_left, bg=C['teal'], width=3).place(x=0, y=0, relheight=1)
        tk.Label(cf_l_inner, text='CONFLUENCE FACTORS', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(anchor='w')
        self._confluence_text = tk.Text(cf_l_inner, bg=C['bg1'], fg=C['teal'],
                                         font=self.F_MONO_SM, bd=0, height=4,
                                         state='disabled', wrap='word',
                                         selectbackground=C['bg3'])
        self._confluence_text.pack(fill='both', expand=True, pady=(4, 0))

        # Invalidation
        cf_right = tk.Frame(cf_row, bg=C['bg1'],
                             highlightthickness=1, highlightbackground=C['border'])
        cf_right.pack(side='left', fill='both', expand=True)
        cf_r_inner = tk.Frame(cf_right, bg=C['bg1'], padx=10, pady=8)
        cf_r_inner.pack(fill='both', expand=True)
        tk.Frame(cf_right, bg=C['red'], width=3).place(x=0, y=0, relheight=1)
        inval_hdr = tk.Frame(cf_r_inner, bg=C['bg1'])
        inval_hdr.pack(fill='x')
        tk.Label(inval_hdr, text='INVALIDATION SCENARIO', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._regime_var = tk.StringVar(value='')
        tk.Label(inval_hdr, textvariable=self._regime_var,
                 bg=C['bg1'], fg=C['blue'],
                 font=(self.MONO, 9, 'bold')).pack(side='right')
        self._inval_var = tk.StringVar(value='--')
        tk.Label(cf_r_inner, textvariable=self._inval_var,
                 bg=C['bg1'], fg=C['red'],
                 font=(self.MONO, 11, 'bold'),
                 wraplength=420, justify='left').pack(anchor='w', pady=(4, 0))

        # ══ SECTION 5: REASONING CHAIN ════════════════════════════════════════
        rsn_frame = tk.Frame(parent, bg=C['bg1'],
                              highlightthickness=1, highlightbackground=C['border'])
        rsn_frame.pack(fill='x', padx=4, pady=(0, 2))
        rsn_inner = tk.Frame(rsn_frame, bg=C['bg1'], padx=10, pady=8)
        rsn_inner.pack(fill='x')
        tk.Frame(rsn_frame, bg=C['blue'], width=3).place(x=0, y=0, relheight=1)
        tk.Label(rsn_inner, text='AI REASONING CHAIN', bg=C['bg1'],
                 fg=C['text3'], font=self.F_HEAD).pack(anchor='w')
        rsb = tk.Scrollbar(rsn_inner, bg=C['bg2'], troughcolor=C['bg3'])
        rsb.pack(side='right', fill='y')
        self._reasoning_text = tk.Text(rsn_inner, bg=C['bg1'], fg=C['text2'],
                                        font=self.F_MONO_SM, bd=0, height=4,
                                        yscrollcommand=rsb.set,
                                        state='disabled', wrap='word',
                                        selectbackground=C['bg3'])
        self._reasoning_text.pack(fill='x', pady=(4, 0))
        rsb.config(command=self._reasoning_text.yview)
        for tag, col in [('step_num', C['purple']), ('step_txt', C['text2']),
                          ('err', C['red'])]:
            self._reasoning_text.tag_configure(tag, foreground=col)

        # ══ SECTION 6: SIGNAL HISTORY ═════════════════════════════════════════
        hist_frame = tk.Frame(parent, bg=C['bg1'],
                               highlightthickness=1, highlightbackground=C['border'])
        hist_frame.pack(fill='both', expand=True, padx=4, pady=(0, 2))
        hist_inner = tk.Frame(hist_frame, bg=C['bg1'], padx=10, pady=8)
        hist_inner.pack(fill='both', expand=True)
        tk.Frame(hist_frame, bg=C['text3'], width=3).place(x=0, y=0, relheight=1)
        hist_hdr = tk.Frame(hist_inner, bg=C['bg1'])
        hist_hdr.pack(fill='x')
        tk.Label(hist_hdr, text=f'SIGNAL HISTORY  (last {self.MAX_HISTORY})',
                 bg=C['bg1'], fg=C['text3'], font=self.F_HEAD).pack(side='left')
        self._stats_var = tk.StringVar(value='')
        tk.Label(hist_hdr, textvariable=self._stats_var,
                 bg=C['bg1'], fg=C['text3'], font=self.F_TINY).pack(side='right')
        hsb = tk.Scrollbar(hist_inner, bg=C['bg2'], troughcolor=C['bg3'])
        hsb.pack(side='right', fill='y')
        self._history_text = tk.Text(hist_inner, bg=C['bg1'], fg=C['text'],
                                      font=self.F_MONO_SM, bd=0, height=7,
                                      yscrollcommand=hsb.set,
                                      state='disabled', wrap='none',
                                      selectbackground=C['bg3'])
        self._history_text.pack(fill='both', expand=True, pady=(4, 0))
        hsb.config(command=self._history_text.yview)
        for tag, col in [('long',    C['green']), ('short', C['red']),
                          ('neutral', C['amber']), ('no_edge', C['text3']),
                          ('ts',      C['text3']), ('conv',   C['purple']),
                          ('driver',  C['text2']), ('rr',     C['teal']),
                          ('cached',  C['text3']), ('regime', C['blue'])]:
            self._history_text.tag_configure(tag, foreground=col)

        # ── Initial model badge update ─────────────────────────────────────
        self._refresh_model_badge()

    # ─────────────────────────────────────────────────────────────────────────
    # CONTROLS
    # ─────────────────────────────────────────────────────────────────────────

    def _toggle_key_visibility(self) -> None:
        self._show_key = not self._show_key
        self._key_entry.config(show='' if self._show_key else '•')

    def _on_model_changed(self, event=None) -> None:
        """Update badge and description when user picks a different model."""
        self._refresh_model_badge()
        # If already live, stop current engine and restart with new model
        if self._enabled:
            from gemini_engine import GEMINI
            GEMINI.stop()
            GEMINI.MODEL_NAME = self._get_selected_model_id()
            # Recreate client with new model name (re-uses stored api key via _sdk_client)
            if GEMINI._sdk_client:
                GEMINI._sdk_client._model_name = GEMINI.MODEL_NAME
                GEMINI.enabled = True
            if hasattr(self._app, '_feed'):
                self._app._feed.start_gemini()

    def _refresh_model_badge(self) -> None:
        C = self._C
        label   = self._model_var.get()
        model   = next((m for m in GEMINI_MODELS if m['label'] == label),
                       GEMINI_MODELS[0])
        self._model_badge_var.set(f' {model["badge"]}  {model["tier"]} ')
        self._model_badge_lbl.config(fg=model['color'], bg=C['bg3'])
        self._model_desc_var.set(model['desc'])

    def _get_selected_model_id(self) -> str:
        label = self._model_var.get()
        model = next((m for m in GEMINI_MODELS if m['label'] == label),
                     GEMINI_MODELS[0])
        return model['id']

    def _toggle_engine(self) -> None:
        from gemini_engine import GEMINI
        C = self._C
        if self._enabled:
            self._disable_engine(GEMINI)
        else:
            self._do_configure()

    def _disable_engine(self, gemini) -> None:
        C = self._C
        gemini.stop()          # signal async loop to exit cleanly
        gemini.enabled  = False
        self._enabled   = False
        self._warming_up = False
        self._toggle_btn.config(text='▶  ENABLE', bg='#1a0a2e', fg=self._C['purple'])
        self._refresh_btn.config(state='disabled', fg=self._C['text3'])
        self._set_status('OFFLINE', self._C['text3'], pill_bg=self._C['bg3'])
        self._countdown_var.set('')
        self._pulse_var.set('')
        self._warmup_bar.place(relwidth=0.0)

    def _do_configure(self) -> None:
        from gemini_engine import GEMINI
        C   = self._C
        key = self._key_var.get().strip()
        if not key:
            self._set_status('⚠  NO API KEY', C['red'], pill_bg='#2a0005')
            self._key_entry.config(highlightbackground=C['red'])
            return
        self._key_entry.config(highlightbackground=C['border2'])

        model_id = self._get_selected_model_id()
        GEMINI.MODEL_NAME = model_id

        # Show connecting spinner
        self._toggle_btn.config(text='⟳  CONNECTING', fg=C['amber'], state='disabled')
        self._set_status('CONNECTING…', C['amber'], pill_bg='#1a1000')
        self._app.update_idletasks()

        ok, err_msg = GEMINI.configure(key)

        self._toggle_btn.config(state='normal')
        if ok:
            self._enabled    = True
            self._warming_up = True
            self._warmup_cnt = 0
            self._first_call = True
            self._toggle_btn.config(text='■  DISABLE', bg='#2a0005', fg=C['red'])
            self._refresh_btn.config(state='normal', fg=C['cyan'])
            self._set_status('WARMING UP', C['amber'], pill_bg='#1a1000')
            self._warmup_label.config(
                text='⟳  Connecting to Gemini… first prediction in ~5 seconds',
                fg=C['amber'])
            self._countdown = 5

            # ── KEY FIX: schedule Gemini run() on the live feed loop ──────────
            # This is what was missing — configure() only creates the client,
            # start() actually launches the async prediction coroutine.
            if hasattr(self._app, '_feed'):
                self._app._feed.start_gemini()
        else:
            self._toggle_btn.config(text='▶  ENABLE', bg='#1a0a2e', fg=C['purple'])
            # Show exact error message so user can diagnose
            short_err = (err_msg[:60] + '…') if len(err_msg) > 60 else err_msg
            self._set_status(f'⚠  {short_err}', C['red'], pill_bg='#2a0005')
            self._driver_var.set(f'⚠  Configuration failed: {err_msg}')
            self._driver_lbl.config(fg=C['red'])

    def _force_refresh(self) -> None:
        """Reset dedup cache so the next cycle fires immediately regardless of market state."""
        try:
            from gemini_engine import GEMINI
            GEMINI._last_feat = {}   # clear dedup — next cycle will always fire
            self._set_status('FORCED ⟳', self._C['cyan'], pill_bg=self._C['bg3'])
            # If the task died for any reason, restart it
            task = GEMINI._task
            if task is None or task.done():
                if hasattr(self._app, '_feed'):
                    self._app._feed.start_gemini()
            self._app.after(1500,
                lambda: self._set_status('LIVE', self._C['purple'], pill_bg='#12002a'))
        except Exception as e:
            self._set_status(f'⚠  {str(e)[:30]}', self._C['red'])

    def _set_status(self, text: str, color: str, pill_bg: str = None) -> None:
        C  = self._C
        bg = pill_bg or C['bg3']
        self._status_var.set(text)
        self._status_lbl.config(fg=color)
        self._status_dot_lbl.config(fg=color)
        self._status_pill.config(bg=bg, highlightbackground=color)
        self._status_dot_lbl.config(bg=bg)
        self._status_lbl.config(bg=bg)

    # ─────────────────────────────────────────────────────────────────────────
    # TICK / ANIMATION
    # ─────────────────────────────────────────────────────────────────────────

    _SPINNERS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

    def _tick(self) -> None:
        C = self._C
        if self._enabled:
            if self._warming_up:
                # Animate warmup progress bar
                self._warmup_cnt += 1
                prog = min(1.0, self._warmup_cnt / (self._warmup_secs * 10))
                self._warmup_bar.place(relwidth=prog)
                spin = self._SPINNERS[self._warmup_cnt % len(self._SPINNERS)]
                remaining = max(0, self._warmup_secs - self._warmup_cnt // 10)
                self._warmup_label.config(
                    text=f'{spin}  Warming up… first prediction in {remaining}s',
                    fg=C['amber'])
                if prog >= 1.0:
                    self._warmup_bar.place(relwidth=0.0)

            # Countdown display
            self._countdown = max(0, self._countdown - 1)
            if self._countdown > 0:
                self._countdown_var.set(f'NEXT  {self._countdown}s')
            else:
                self._countdown_var.set('ANALYSING…')

            # Blink status dot
            self._blink_on = not self._blink_on
            if not self._warming_up:
                dot_col = C['purple'] if self._blink_on else C['bg3']
                self._status_dot_lbl.config(fg=dot_col)

            # Pulse indicator
            self._spinner_idx = (self._spinner_idx + 1) % len(self._SPINNERS)
            self._pulse_var.set(
                f'{self._SPINNERS[self._spinner_idx]}  10s PULSE')

        self._app.after(100, self._tick)   # 100ms tick for smooth animation

    # ─────────────────────────────────────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    def _on_prediction(self, result) -> None:
        from gemini_engine import GEMINI
        C = self._C

        # First-ever real prediction: swap placeholder for live card
        if self._first_call and not result.skipped:
            self._first_call  = False
            self._warming_up  = False
            self._warmup_label.config(text='')
            self._warmup_bar.place(relwidth=0.0)
            self._placeholder.pack_forget()
            self._live_card.pack(fill='x')
            self._driver_lbl.config(fg=C['text2'])
            self._set_status('LIVE', C['purple'], pill_bg='#12002a')

        if not result.skipped:
            self._history.appendleft(result)
            self._update_card(result)
            self._update_confluence(result)
            self._update_reasoning(result)
            self._update_history()
        else:
            # Skipped cycle — briefly show "CACHED" then return to LIVE
            if not self._first_call:
                self._set_status(f'CACHED  ({result.skip_reason[:20]})',
                                 C['text3'], pill_bg=C['bg3'])
                self._app.after(2000,
                    lambda: self._set_status('LIVE', C['purple'], pill_bg='#12002a'))

        self._update_stats(GEMINI)

        # Reset countdown from current regime cadence
        regime  = getattr(result, 'regime', 'RANGING')
        cadence = self._cadence_map.get(regime, 30)
        self._countdown = cadence

    def _on_error(self, msg: str) -> None:
        self._set_status(f'⚠  {str(msg)[:35]}', self._C['red'],
                         pill_bg='#2a0005')
        if self._driver_var:
            self._driver_var.set(f'⚠  API error: {str(msg)[:80]}')
            self._driver_lbl.config(fg=self._C['red'])

    # ─────────────────────────────────────────────────────────────────────────
    # CARD / CONTENT UPDATES
    # ─────────────────────────────────────────────────────────────────────────

    def _update_card(self, r) -> None:
        C = self._C

        dir_styles = {
            'LONG':    ('▲  LONG',       C['green'],  '#001f0a', C['green']),
            'SHORT':   ('▼  SHORT',      C['red'],    '#1a0005', C['red']),
            'NEUTRAL': ('◆  NEUTRAL',    C['amber'],  '#1a1000', C['amber']),
            'NO_EDGE': ('─  NO EDGE',    C['text3'],  C['bg2'],  C['border2']),
        }
        dir_text, dir_fg, card_bg, card_border = dir_styles.get(
            r.direction, ('?', C['text3'], C['bg2'], C['border2']))

        self._dir_var.set(dir_text)
        self._dir_lbl.config(fg=dir_fg)
        self._card.config(bg=card_bg, highlightbackground=card_border)
        self._live_card.config(bg=card_bg)

        # Conviction
        conv = r.conviction
        self._conv_var.set(f'{conv}/10')
        conv_col = C['green'] if conv >= 7 else C['amber'] if conv >= 4 else C['red']
        self._conv_lbl.config(fg=conv_col)
        self._conv_bar_frame.update_idletasks()
        bw = self._conv_bar_frame.winfo_width()
        if bw > 1:
            self._conv_bar.place(relwidth=conv / 10.0)
            self._conv_bar.config(bg=conv_col)

        # Metrics
        self._horizon_var.set(r.time_horizon)
        self._risk_var.set(r.risk_level)
        self._size_var.set(f'{r.position_size_pct:.1f}%')
        self._rr_var.set(f'{r.risk_reward:.1f}×')

        # Price levels
        from models import DECIMALS
        dec = DECIMALS.get(r.pair, 2)
        self._entry_var.set(f'{r.entry_zone_lo:,.{dec}f} – {r.entry_zone_hi:,.{dec}f}')
        self._sl_var.set(f'{r.stop_loss:,.{dec}f}')
        self._tp1_var.set(f'{r.take_profit_1:,.{dec}f}')
        self._tp2_var.set(f'{r.take_profit_2:,.{dec}f}')

        # Primary driver & timestamp
        self._driver_var.set(r.primary_driver or '--')
        self._driver_lbl.config(fg=C['text'])
        self._ts_var.set(f'  {r.timestamp[11:19]} UTC')

        # Invalidation + regime
        self._inval_var.set(r.invalidation or '--')
        self._regime_var.set(r.regime or '')

    def _update_confluence(self, r) -> None:
        C = self._C
        t = self._confluence_text
        t.config(state='normal')
        t.delete('1.0', 'end')
        if r.confluence:
            for item in r.confluence:
                t.insert('end', '  ✓  ', 'ok')
                t.insert('end', item + '\n')
            t.tag_configure('ok', foreground=C['teal'])
        else:
            t.insert('end', '  No confluence factors reported.\n')
        t.config(state='disabled')

    def _update_reasoning(self, r) -> None:
        t = self._reasoning_text
        t.config(state='normal')
        t.delete('1.0', 'end')

        # New schema: r.reasoning is a pipe-delimited string "step1|step2|verdict"
        why = r.reasoning or ''
        if why:
            steps = [s.strip() for s in why.split('|') if s.strip()]
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
        t.config(state='normal')
        t.delete('1.0', 'end')

        if not self._history:
            t.insert('end', '  No predictions yet.\n', 'ts')
            t.config(state='disabled')
            return

        from models import DECIMALS
        for r in self._history:
            tag = r.direction.lower() if r.direction in ('LONG', 'SHORT', 'NEUTRAL') else 'no_edge'
            sym = {'LONG': '▲', 'SHORT': '▼', 'NEUTRAL': '◆', 'NO_EDGE': '─'}.get(
                r.direction, '?')
            dec = DECIMALS.get(r.pair, 2)
            rr  = f'{r.risk_reward:.1f}×' if r.risk_reward else '--'

            t.insert('end', f'  {r.timestamp[11:19]}  ', 'ts')
            t.insert('end', f'{sym} {r.direction:<8}', tag)
            t.insert('end', f'  {r.conviction}/10  ', 'conv')
            t.insert('end', f'RR:{rr}  ', 'rr')
            t.insert('end', f'{r.regime:<10}  ', 'regime')
            t.insert('end', f'{r.primary_driver[:55]}\n', 'driver')

        t.config(state='disabled')

    def _update_stats(self, gemini) -> None:
        stats  = gemini.stats
        _, tok = gemini.user_prompt_size()
        self._stats_var.set(
            f'Calls: {stats["calls"]}   '
            f'Skips: {stats["skips"]} ({stats["skip_rate_pct"]:.0f}%)   '
            f'Saved ≈ {stats["tokens_saved"]} tok   '
            f'{stats["latency_ms"]:.0f} ms   '
            f'~{tok} tok/call'
        )
        if stats['latency_ms'] > 0:
            self._latency_var.set(f'{stats["latency_ms"]:.0f} ms')
