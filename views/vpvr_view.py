"""
HPAT v6 — views/vpvr_view.py
VPVR (Volume Profile Visible Range) panel.
Object-pooled canvas — no delete('all') ever.
"""

from __future__ import annotations
import tkinter as tk
from typing import List

from views.base_view import BaseView
from analytics_engine import get_poc, fmt, fmt_k


class VPVRView(BaseView):

    _MAX_ROWS = 20

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()
        self._init_ui(parent)
        self._init_pools()

    def _init_ui(self, parent: tk.Widget) -> None:
        C = self._C
        pf, body = self._panel(parent, 'VPVR — VOLUME PROFILE', 'POC')
        pf.pack(fill='x', pady=(0, 1))

        self.poc_var = tk.StringVar(value='--')
        self.sig_var = tk.StringVar(value='Awaiting volume data...')
        row = tk.Frame(body, bg=C['bg1'])
        row.pack(fill='x', pady=(0, 3))
        tk.Label(row, text='POC:', bg=C['bg1'],
                 fg=C['text3'], font=self.F_MONO_SM).pack(side='left')
        tk.Label(row, textvariable=self.poc_var, bg=C['bg1'],
                 fg=C['amber'], font=(self.MONO, 12, 'bold')).pack(side='left', padx=6)
        tk.Label(body, textvariable=self.sig_var, bg=C['bg1'],
                 fg=C['text2'], font=self.F_MONO_SM, wraplength=280).pack(anchor='w', pady=2)

        self.canvas = tk.Canvas(body, bg=C['bg1'], bd=0,
                                 highlightthickness=0, height=160)
        self.canvas.pack(fill='x', pady=3)

    def _init_pools(self) -> None:
        n      = self._MAX_ROWS
        canvas = self.canvas
        C      = self._C
        font   = self.F_TINY

        self._bg_rects: List[int]    = [canvas.create_rectangle(0, 0, 0, 0, fill=C['bg1'], outline='') for _ in range(n)]
        self._bar_rects: List[int]   = [canvas.create_rectangle(0, 0, 0, 0, fill=C['green'], outline='') for _ in range(n)]
        self._price_texts: List[int] = [canvas.create_text(0, 0, text='', font=font, fill=C['text2'], anchor='e') for _ in range(n)]
        self._vol_texts: List[int]   = [canvas.create_text(0, 0, text='', font=font, fill=C['text3'], anchor='e') for _ in range(n)]
        self._poc_texts: List[int]   = [canvas.create_text(0, 0, text='', font=font, fill=C['amber'], anchor='w') for _ in range(n)]

    def update(self, snapshot) -> None:
        if not snapshot.vpvr or not snapshot.price:
            return

        C      = self._C
        canvas = self.canvas
        poc, dist = get_poc()

        self.poc_var.set(fmt(poc, 0) if poc else '--')
        if poc:
            self.sig_var.set(
                f'POC {fmt(poc, 0)} | dist {dist:.2f}% | '
                f'{"▲ ABOVE POC" if snapshot.price > poc else "▼ BELOW POC"}'
            )

        canvas.update_idletasks()
        cw      = canvas.winfo_width() or 260
        row_h   = 15
        label_w = 65
        bar_max = cw - label_w - 55

        # Pick the 16 bins closest to current price
        nearby = sorted(
            snapshot.vpvr.items(),
            key=lambda kv: abs(kv[0] - snapshot.price)
        )[:16]
        nearby.sort(key=lambda kv: kv[0], reverse=True)

        max_v   = max(v for _, v in nearby) if nearby else 1.0
        row_idx = 0
        y       = 2

        for b, v in nearby:
            pct     = max(3, int(v / max_v * bar_max))
            is_poc  = abs(b - poc) < 0.001 if poc else False
            col_p   = C['amber'] if is_poc else (C['red'] if b > snapshot.price else C['green'])

            # BG highlight for POC
            canvas.coords(self._bg_rects[row_idx], 0, y, cw, y + row_h - 1)
            canvas.itemconfig(self._bg_rects[row_idx],
                               fill=C['bg3'] if is_poc else C['bg1'])

            # Bar
            canvas.coords(self._bar_rects[row_idx],
                           label_w, y + 2, label_w + pct, y + row_h - 2)
            canvas.itemconfig(self._bar_rects[row_idx], fill=col_p)

            # Price label
            canvas.coords(self._price_texts[row_idx], label_w - 2, y + row_h // 2)
            canvas.itemconfig(self._price_texts[row_idx],
                               text=fmt(b, 0), fill=col_p)

            # Volume value (far right)
            canvas.coords(self._vol_texts[row_idx], cw - 2, y + row_h // 2)
            canvas.itemconfig(self._vol_texts[row_idx], text=fmt_k(v))

            # POC label
            canvas.coords(self._poc_texts[row_idx], label_w + pct + 4, y + row_h // 2)
            canvas.itemconfig(self._poc_texts[row_idx],
                               text='POC' if is_poc else '')

            row_idx += 1
            y       += row_h

        # Hide unused pool slots
        for i in range(row_idx, self._MAX_ROWS):
            for pool in (self._bg_rects, self._bar_rects):
                canvas.coords(pool[i], 0, 0, 0, 0)
            for pool in (self._price_texts, self._vol_texts, self._poc_texts):
                canvas.coords(pool[i], 0, 0)
                canvas.itemconfig(pool[i], text='')

        canvas.config(height=y + 2)
