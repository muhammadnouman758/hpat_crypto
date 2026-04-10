"""
HPAT v6 — views/orderbook_view.py
Order-book heatmap rendered with object pooling.
Canvas rectangles and text labels are created once at init,
then updated via coords() / itemconfig() on each tick — no delete('all').
"""

from __future__ import annotations
import tkinter as tk
from typing import List

from views.base_view import BaseView
from analytics_engine import calc_obi, fmt


class OrderBookView(BaseView):
    """Order-book heatmap panel with object-pooled canvas rendering."""

    _OB_ROWS = 13   # 6 asks + divider + 6 bids

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()
        self._init_ui(parent)
        self._init_pools()

    def _init_ui(self, parent: tk.Widget) -> None:
        C = self._C
        pf, body = self._panel(parent, 'ORDER BOOK HEATMAP')
        pf.pack(fill='both', expand=True, pady=(0, 0))
        self._pf = pf

        self.obi_var = tk.StringVar(value='OBI: --')
        tk.Label(body, textvariable=self.obi_var, bg=C['bg1'],
                 fg=C['cyan'], font=(self.MONO, 12, 'bold')).pack(anchor='w', pady=(0, 4))

        self.canvas = tk.Canvas(body, bg=C['bg1'], bd=0,
                                 highlightthickness=0,
                                 height=self._OB_ROWS * 17 + 4)
        self.canvas.pack(fill='x')
        self._body = body

    def _init_pools(self) -> None:
        """Pre-create all canvas items so we never call delete('all')."""
        n       = self._OB_ROWS
        canvas  = self.canvas
        C       = self._C
        font    = self.F_TINY

        # Background highlight rectangles (1 per row)
        self._bg_rects: List[int] = [
            canvas.create_rectangle(0, 0, 0, 0, fill=C['bg1'], outline='')
            for _ in range(n)
        ]
        # Volume bars (1 per row)
        self._bar_rects: List[int] = [
            canvas.create_rectangle(0, 0, 0, 0, fill=C['red'], outline='')
            for _ in range(n)
        ]
        # Price labels (1 per row)
        self._price_texts: List[int] = [
            canvas.create_text(0, 0, text='', font=font,
                                fill=C['text2'], anchor='e')
            for _ in range(n)
        ]
        # Volume labels (1 per row)
        self._vol_texts: List[int] = [
            canvas.create_text(0, 0, text='', font=font,
                                fill=C['text3'], anchor='w')
            for _ in range(n)
        ]
        # Wall indicator texts (1 per row)
        self._wall_texts: List[int] = [
            canvas.create_text(0, 0, text='', font=font,
                                fill=C['amber'], anchor='w')
            for _ in range(n)
        ]
        # Mid-price divider line
        self._mid_line: int = canvas.create_line(0, 0, 0, 0,
                                                   fill=C['border2'], width=1)
        # Mid-price text
        self._mid_text: int = canvas.create_text(0, 0, text='',
                                                   font=self.F_MONO_SM,
                                                   fill=C['cyan'], anchor='center')

    def update(self, snapshot) -> None:
        """Called from the UI update loop with a state snapshot."""
        bids = list(snapshot.ob_bids)[:8]
        asks = list(snapshot.ob_asks)[:8]
        if not bids or not asks:
            return

        obi = calc_obi(bids, asks)
        self.obi_var.set(
            f'OBI: {obi:+.3f}  '
            f'{"▲ BID HEAVY" if obi > 0.2 else "▼ ASK HEAVY" if obi < -0.2 else "BALANCED"}')

        C       = self._C
        canvas  = self.canvas
        canvas.update_idletasks()
        cw      = canvas.winfo_width() or 280
        row_h   = 17
        label_w = 82
        bar_max = cw - label_w - 55

        all_q   = [float(x[1]) for x in bids + asks]
        max_q   = max(all_q) if all_q else 1.0

        levels_asks = list(reversed(asks[:6]))
        levels_bids = bids[:6]
        mid_price   = snapshot.price
        from models import DECIMALS
        dec         = DECIMALS.get(snapshot.pair, 2)

        row_idx = 0
        y       = 2

        # — ASK rows ——————————————————————————————————————————————————————————
        for level in levels_asks:
            pr  = float(level[0]); q = float(level[1])
            intensity = q / max_q
            is_wall   = intensity > 0.6
            bar_w     = max(4, int(intensity * bar_max))
            bar_col   = C['amber'] if is_wall else C['red']

            # BG tint
            alpha_bg = int(intensity * 30)
            hex_bg   = f'#{alpha_bg:02x}0000'
            canvas.coords(self._bg_rects[row_idx], 0, y, cw, y + row_h - 1)
            canvas.itemconfig(self._bg_rects[row_idx], fill=hex_bg)

            # Bar
            canvas.coords(self._bar_rects[row_idx],
                           label_w, y + 3, label_w + bar_w, y + row_h - 3)
            canvas.itemconfig(self._bar_rects[row_idx], fill=bar_col)

            # Price
            canvas.coords(self._price_texts[row_idx], label_w - 4, y + row_h // 2)
            canvas.itemconfig(self._price_texts[row_idx],
                               text=fmt(pr, dec), fill=C['red'])

            # Volume
            canvas.coords(self._vol_texts[row_idx],
                           label_w + bar_w + 6, y + row_h // 2)
            canvas.itemconfig(self._vol_texts[row_idx], text=f'{q:.3f}')

            # Wall marker
            canvas.coords(self._wall_texts[row_idx],
                           cw - 4, y + row_h // 2)
            canvas.itemconfig(self._wall_texts[row_idx],
                               text='⚡WALL' if is_wall else '',
                               anchor='e')

            row_idx += 1
            y       += row_h

        # — Divider ───────────────────────────────────────────────────────────
        canvas.coords(self._mid_line, 0, y, cw, y)
        canvas.coords(self._mid_text, cw // 2, y + row_h // 2)
        canvas.itemconfig(self._mid_text, text=fmt(mid_price, dec) if mid_price else '--')
        y += row_h

        # — BID rows ──────────────────────────────────────────────────────────
        for level in levels_bids:
            pr  = float(level[0]); q = float(level[1])
            intensity = q / max_q
            is_wall   = intensity > 0.6
            bar_w     = max(4, int(intensity * bar_max))
            bar_col   = C['amber'] if is_wall else C['green']

            alpha_bg = int(intensity * 30)
            hex_bg   = f'#00{alpha_bg:02x}00'
            canvas.coords(self._bg_rects[row_idx], 0, y, cw, y + row_h - 1)
            canvas.itemconfig(self._bg_rects[row_idx], fill=hex_bg)

            canvas.coords(self._bar_rects[row_idx],
                           label_w, y + 3, label_w + bar_w, y + row_h - 3)
            canvas.itemconfig(self._bar_rects[row_idx], fill=bar_col)

            canvas.coords(self._price_texts[row_idx], label_w - 4, y + row_h // 2)
            canvas.itemconfig(self._price_texts[row_idx],
                               text=fmt(pr, dec), fill=C['green'])

            canvas.coords(self._vol_texts[row_idx],
                           label_w + bar_w + 6, y + row_h // 2)
            canvas.itemconfig(self._vol_texts[row_idx], text=f'{q:.3f}')

            canvas.coords(self._wall_texts[row_idx],
                           cw - 4, y + row_h // 2)
            canvas.itemconfig(self._wall_texts[row_idx],
                               text='⚡WALL' if is_wall else '',
                               anchor='e')

            row_idx += 1
            y       += row_h

        # Hide unused pool slots
        for i in range(row_idx, len(self._bg_rects)):
            canvas.coords(self._bg_rects[i],   0, 0, 0, 0)
            canvas.coords(self._bar_rects[i],  0, 0, 0, 0)
            canvas.coords(self._price_texts[i], 0, 0)
            canvas.itemconfig(self._price_texts[i], text='')
            canvas.coords(self._vol_texts[i],   0, 0)
            canvas.itemconfig(self._vol_texts[i], text='')
            canvas.coords(self._wall_texts[i],  0, 0)
            canvas.itemconfig(self._wall_texts[i], text='')

        # Resize canvas height to fit actual content
        canvas.config(height=y + 2)
