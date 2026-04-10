"""
HPAT v6 — views/base_view.py
Base class for all decoupled UI view components.
Each view subscribes to events via the EventBus.
Canvas rendering uses object pooling — create once, update coordinates.
"""

from __future__ import annotations
import tkinter as tk
from typing import Optional, Callable
import sys


class BaseView:
    """
    Lightweight base for a Tkinter UI component that:
      - Owns a root tk.Frame
      - Can subscribe to BUS events (callbacks scheduled via tk.after)
      - Provides object-pool helpers for Canvas items
    """

    def __init__(self, parent: tk.Widget, app: tk.Tk,
                 bg: Optional[str] = None) -> None:
        from models import C, BUS
        self._C   = C
        self._BUS = BUS
        self._app = app        # root Tk window (for .after scheduling)
        self.bg   = bg or C['bg1']
        self.frame = tk.Frame(parent, bg=self.bg)
        self._subscriptions: list = []

    # ── Event subscription helpers ────────────────────────────────────────────

    def subscribe(self, kind: str, callback: Callable) -> None:
        """Subscribe to an event kind. Callback runs on the feed thread —
        use _schedule() to touch Tk widgets safely."""
        self._BUS.subscribe(kind, callback)
        self._subscriptions.append((kind, callback))

    def _schedule(self, callback: Callable, *args) -> None:
        """Schedule a callback on the Tk main thread."""
        self._app.after(0, callback, *args)

    # ── Panel / label builders (shared helpers) ───────────────────────────────

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

    def _panel(self, parent: tk.Widget, title: str,
               right_text: str = '', accent: Optional[str] = None):
        C = self._C
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

    def _stat_row(self, parent: tk.Widget, label: str, color: Optional[str] = None):
        """Returns (StringVar, Label) for a key/value display row."""
        C = self._C
        r = tk.Frame(parent, bg=C['bg1'])
        r.pack(fill='x', pady=1)
        tk.Label(r, text=label, bg=C['bg1'], fg=C['text3'],
                 font=self.F_LABEL, width=16, anchor='w').pack(side='left')
        v = tk.StringVar(value='--')
        l = tk.Label(r, textvariable=v, bg=C['bg1'], fg=color or C['text'],
                     font=self.F_MONO_SM, anchor='e')
        l.pack(side='right')
        return v, l

    # ── Canvas object pool helpers ────────────────────────────────────────────

    @staticmethod
    def pool_rects(canvas: tk.Canvas, pool: list, count: int,
                   fill: str = '#000', outline: str = '') -> list:
        """
        Ensure `pool` has at least `count` rectangle canvas items.
        Creates new items as needed; hides excess items.
        Returns the pool list.
        """
        while len(pool) < count:
            pool.append(canvas.create_rectangle(0, 0, 0, 0,
                                                 fill=fill, outline=outline))
        # Hide all first; caller will show the ones it needs
        for i, item in enumerate(pool):
            if i >= count:
                canvas.coords(item, 0, 0, 0, 0)
        return pool

    @staticmethod
    def pool_texts(canvas: tk.Canvas, pool: list, count: int,
                   font=None, fill: str = '#607898') -> list:
        """
        Ensure `pool` has at least `count` text canvas items.
        """
        while len(pool) < count:
            pool.append(canvas.create_text(0, 0, text='', font=font or ('Courier', 9),
                                            fill=fill, anchor='e'))
        for i, item in enumerate(pool):
            if i >= count:
                canvas.itemconfig(item, text='')
        return pool
