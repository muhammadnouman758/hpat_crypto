"""
HPAT v6 — views/account_view.py
Account dashboard tab: balances, P&L, trade history, open orders.
Subscribes to the 'acc_refresh' event rather than polling.
All monetary display uses Decimal.
Credentials loaded from env — UI shows status only, no key entry.
"""

from __future__ import annotations
import datetime
import os
import tkinter as tk
from decimal import Decimal
from typing import Optional

from views.base_view import BaseView
from models import ACC, BUS
from analytics_engine import fmt


class AccountView(BaseView):

    def __init__(self, parent: tk.Widget, app: tk.Tk) -> None:
        super().__init__(parent, app)
        self._build_fonts()
        self._acc_filter = 'ALL'
        self._build_ui(parent)
        # Subscribe to refresh events — no polling
        self.subscribe('acc_refresh', lambda e: self._schedule(self._update_ui))

    def _build_ui(self, parent: tk.Widget) -> None:
        C = self._C
        tab = parent
        top = tk.Frame(tab, bg=C['bg1'])
        top.pack(fill='x', padx=4, pady=4)

        # ── Status bar (no API key entry — creds come from env) ───────────────
        pf, body = self._panel(top, 'ACCOUNT CONNECT')
        pf.pack(side='left', fill='y', padx=(0, 4))

        self.status_var = tk.StringVar(value='DISCONNECTED')
        self.status_lbl = tk.Label(body, textvariable=self.status_var,
                                    bg=C['bg1'], fg=C['red'],
                                    font=(self.MONO, 11, 'bold'))
        self.status_lbl.pack(anchor='w')

        env_note = tk.Label(
            body,
            text='Keys loaded from BINANCE_API_KEY / BINANCE_API_SECRET env vars\n'
                 'or from a .env file in the working directory.',
            bg=C['bg1'], fg=C['text3'], font=self.F_TINY,
            justify='left',
        )
        env_note.pack(anchor='w', pady=2)

        self.connect_btn = tk.Button(
            body, text='CONNECT / REFRESH',
            bg=C['bg3'], fg=C['cyan'],
            font=self.F_MONO_SM, bd=0, padx=12, pady=4,
            cursor='hand2',
            command=self._connect,
        )
        self.connect_btn.pack(anchor='w', pady=(4, 0))

        self.lastrefresh_var = tk.StringVar(value='')
        tk.Label(body, textvariable=self.lastrefresh_var,
                 bg=C['bg1'], fg=C['text3'],
                 font=self.F_TINY).pack(anchor='w')

        # ── Summary stats ─────────────────────────────────────────────────────
        pf2, body2 = self._panel(top, 'SUMMARY')
        pf2.pack(side='left', fill='y', padx=(0, 4))

        self.total_var   = tk.StringVar(value='$0.00')
        self.pnl_var     = tk.StringVar(value='$0.00')
        self.wins_var    = tk.StringVar(value='0')
        self.loss_var    = tk.StringVar(value='0')
        self.winrate_var = tk.StringVar(value='0.0%')

        for lbl, var, col in [('TOTAL', self.total_var, C['cyan']),
                                ('PNL',   self.pnl_var,   C['green']),
                                ('WINS',  self.wins_var,  C['green']),
                                ('LOSSES',self.loss_var,  C['red']),
                                ('WIN%',  self.winrate_var, C['amber'])]:
            r = tk.Frame(body2, bg=C['bg1'])
            r.pack(fill='x', pady=1)
            tk.Label(r, text=lbl, bg=C['bg1'], fg=C['text3'],
                     font=self.F_LABEL, width=8, anchor='w').pack(side='left')
            tk.Label(r, textvariable=var, bg=C['bg1'], fg=col,
                     font=self.F_MONO_SM, anchor='e').pack(side='right')

        # Win-rate bar
        self.wr_bar_frame = tk.Frame(body2, bg=C['bg3'], height=8)
        self.wr_bar_frame.pack(fill='x', pady=2)
        self.wr_bar = tk.Frame(self.wr_bar_frame, bg=C['green'], height=8)

        # ── Notebook sub-tabs ─────────────────────────────────────────────────
        from tkinter import ttk
        nb = ttk.Notebook(tab)
        style = ttk.Style()
        style.configure('Acc.TNotebook',     background=C['bg1'], borderwidth=0)
        style.configure('Acc.TNotebook.Tab', background=C['bg2'], foreground=C['text2'],
                        padding=[10, 4], font=self.F_HEAD)
        style.map('Acc.TNotebook.Tab',
                  background=[('selected', C['bg3'])],
                  foreground=[('selected', C['cyan'])])
        nb.configure(style='Acc.TNotebook')
        nb.pack(fill='both', expand=True, padx=4, pady=2)

        # Balances tab
        bal_tab = tk.Frame(nb, bg=C['bg1'])
        nb.add(bal_tab, text='Balances')
        self.bal_text = self._make_text(bal_tab)

        # P&L breakdown tab
        pnl_tab = tk.Frame(nb, bg=C['bg1'])
        nb.add(pnl_tab, text='P&L Breakdown')
        self.pnl_text = self._make_text(pnl_tab)
        self.pnl_text.tag_configure('pos', foreground=C['green'])
        self.pnl_text.tag_configure('neg', foreground=C['red'])
        self.pnl_text.tag_configure('sym', foreground=C['cyan'])
        self.pnl_text.tag_configure('cnt', foreground=C['text3'])

        # Trade history tab
        hist_tab = tk.Frame(nb, bg=C['bg1'])
        nb.add(hist_tab, text='Trade History')
        # Filter row
        frow = tk.Frame(hist_tab, bg=C['bg1'])
        frow.pack(fill='x', pady=2, padx=4)
        tk.Label(frow, text='Filter:', bg=C['bg1'],
                 fg=C['text3'], font=self.F_LABEL).pack(side='left')
        from models import PAIRS, PAIR_LABELS
        for sym in ['ALL'] + [PAIR_LABELS[p] for p in PAIRS]:
            tk.Button(frow, text=sym, bg=C['bg2'], fg=C['text2'],
                      font=self.F_TINY, bd=0, padx=6, pady=2,
                      cursor='hand2',
                      command=lambda s=sym: self._filter(s)).pack(side='left', padx=1)
        self.hist_text = self._make_text(hist_tab)
        for tag, col in [('buy', C['green']), ('sell', C['red']), ('sym', C['cyan']),
                          ('ts', C['text3']), ('price', C['text']), ('qty', C['text2']),
                          ('pnl_pos', C['green']), ('pnl_neg', C['red'])]:
            self.hist_text.tag_configure(tag, foreground=col)

        # Open orders tab
        ord_tab = tk.Frame(nb, bg=C['bg1'])
        nb.add(ord_tab, text='Open Orders')
        self.orders_text = self._make_text(ord_tab)
        for tag, col in [('buy', C['green']), ('sell', C['red']),
                          ('sym', C['cyan']), ('ts', C['text3']),
                          ('price', C['text'])]:
            self.orders_text.tag_configure(tag, foreground=col)

    def _make_text(self, parent: tk.Widget) -> tk.Text:
        C = self._C
        frame = tk.Frame(parent, bg=C['bg1'])
        frame.pack(fill='both', expand=True)
        sb = tk.Scrollbar(frame, bg=C['bg2'], troughcolor=C['bg3'])
        sb.pack(side='right', fill='y')
        t = tk.Text(frame, bg=C['bg1'], fg=C['text'],
                    font=self.F_MONO_SM, bd=0,
                    yscrollcommand=sb.set,
                    state='disabled', wrap='none',
                    selectbackground=C['bg3'])
        t.pack(fill='both', expand=True)
        sb.config(command=t.yview)
        return t

    # ── Connect / refresh ─────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Load credentials from environment and trigger account refresh."""
        self._load_env_credentials()
        if not ACC.api_key:
            self.status_var.set('NO API KEY — set BINANCE_API_KEY env var')
            self.status_lbl.config(fg=self._C['red'])
            return
        self.status_var.set('CONNECTING...')
        self.status_lbl.config(fg=self._C['amber'])
        self.connect_btn.config(state='disabled')
        # Use the feed controller's async refresh
        from data_feed import FeedController
        # Access via the app reference set during construction
        if hasattr(self._app, '_feed'):
            self._app._feed.run_account_refresh()
        else:
            self.status_var.set('Feed not ready')
        self.connect_btn.config(state='normal')

    @staticmethod
    def _load_env_credentials() -> None:
        """Load API keys from environment variables or .env file — never from UI."""
        # Try python-dotenv if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        ACC.api_key    = os.environ.get('BINANCE_API_KEY', 'aE5fetrUhYsrO6xJLMvwqS3OdQq8iRUDDG4yltzSqRCWDaUuBXq6zmbNbrAm8ZT4')
        ACC.api_secret = os.environ.get('BINANCE_API_SECRET', 'jxpFCWe8BZwBlIL2Xp3vtoMf5y2KTrHBWe2sEaQvzqyrtJ3C6oQ2xn2ZO6AUdPQP')

    # ── UI updates ────────────────────────────────────────────────────────────

    def _update_ui(self) -> None:
        C = self._C
        ts = (datetime.datetime.utcfromtimestamp(ACC.last_refresh)
              .strftime('%H:%M:%S UTC') if ACC.last_refresh else '')
        self.lastrefresh_var.set(f'Last: {ts}')

        if ACC.connected:
            self.status_var.set('LIVE')
            self.status_lbl.config(fg=C['green'])
        else:
            err = (ACC.error[:70] if ACC.error else 'Not connected')
            self.status_var.set(f'ERROR: {err}')
            self.status_lbl.config(fg=C['red'])

        self.total_var.set(f'${ACC.total_usd:,.2f}')
        pnl     = ACC.total_pnl
        pnl_str = ('+' if pnl >= 0 else '') + f'${pnl:,.2f}'
        self.pnl_var.set(pnl_str)
        self.wins_var.set(str(ACC.win_trades))
        self.loss_var.set(str(ACC.loss_trades))
        total_t = ACC.win_trades + ACC.loss_trades
        wr = ACC.win_trades / max(total_t, 1) * 100
        self.winrate_var.set(f'{wr:.1f}%')
        self.wr_bar_frame.update_idletasks()
        bw = self.wr_bar_frame.winfo_width()
        if bw > 1:
            self.wr_bar.place(x=0, y=0, relheight=1.0, width=int(bw * wr / 100))
            self.wr_bar.config(bg=C['green'] if wr >= 50 else C['red'])

        self._update_balances()
        self._update_pnl()
        self._update_history()
        self._update_orders()

    def _update_balances(self) -> None:
        t = self.bal_text
        t.config(state='normal'); t.delete('1.0', 'end')
        C = self._C
        t.tag_configure('asset',  foreground=C['cyan'])
        t.tag_configure('free',   foreground=C['green'])
        t.tag_configure('locked', foreground=C['amber'])
        t.tag_configure('usd',    foreground=C['text'])
        if not ACC.balances:
            t.insert('end', '  No balances found.\n')
        else:
            for b in ACC.balances:
                t.insert('end', b.asset.ljust(6),   'asset')
                t.insert('end', f'{b.free:.6f}'.rjust(16), 'free')
                t.insert('end', f'{b.locked:.6f}'.rjust(14), 'locked')
                t.insert('end', f'${b.usd_val:,.2f}'.rjust(14) + '\n', 'usd')
        t.config(state='disabled')

    def _update_pnl(self) -> None:
        t = self.pnl_text
        t.config(state='normal'); t.delete('1.0', 'end')
        if not ACC.realized_pnl:
            t.insert('end', '  No trade history.\n')
        else:
            sym_counts = {}
            for trade in ACC.trade_history:
                if trade.pnl is not None:
                    sym_counts[trade.symbol] = sym_counts.get(trade.symbol, 0) + 1
            sorted_pnl = sorted(ACC.realized_pnl.items(), key=lambda x: x[1], reverse=True)
            max_abs    = max(abs(v) for v in ACC.realized_pnl.values()) or Decimal('1')
            for sym, pnl in sorted_pnl:
                pnl_tag = 'pos' if pnl >= 0 else 'neg'
                pnl_s   = ('+' if pnl >= 0 else '') + f'${pnl:,.2f}'
                bar_len = min(18, int(abs(pnl) / max_abs * 18))
                t.insert('end', sym.replace('USDT', '').ljust(7), 'sym')
                t.insert('end', f'{sym_counts.get(sym, 0)} tx'.rjust(6), 'cnt')
                t.insert('end', f'  {pnl_s:>12}  ', pnl_tag)
                t.insert('end', '▮' * bar_len + '\n', pnl_tag)
        t.config(state='disabled')

    def _update_history(self) -> None:
        t = self.hist_text
        t.config(state='normal'); t.delete('1.0', 'end')
        trades = list(reversed(ACC.trade_history))
        if self._acc_filter != 'ALL':
            sym_full = self._acc_filter + 'USDT'
            trades   = [tr for tr in trades if tr.symbol == sym_full]
        if not trades:
            t.insert('end', '  No trades found.\n')
        else:
            for tr in trades[:200]:
                side_tag = 'buy' if tr.side == 'BUY' else 'sell'
                pnl_tag  = ('pnl_pos' if (tr.pnl or Decimal(0)) >= 0 else 'pnl_neg') if tr.pnl is not None else 'ts'
                pnl_s    = (('+$' if (tr.pnl or 0) >= 0 else '-$') + f'{abs(tr.pnl):,.2f}') if tr.pnl is not None else '--'
                t.insert('end', tr.time.ljust(12),             'ts')
                t.insert('end', tr.symbol.replace('USDT', '').rjust(6) + ' ', 'sym')
                t.insert('end', tr.side.rjust(5) + ' ',        side_tag)
                t.insert('end', f'{tr.price:,.4f}'.rjust(14),  'price')
                t.insert('end', f'{tr.qty:.4f}'.rjust(11),     'qty')
                t.insert('end', f'${tr.quoteQty:,.2f}'.rjust(12), 'price')
                t.insert('end', pnl_s.rjust(13) + '\n',        pnl_tag)
        t.config(state='disabled')

    def _update_orders(self) -> None:
        t = self.orders_text
        t.config(state='normal'); t.delete('1.0', 'end')
        if not ACC.open_orders:
            msg = '  No open orders.' if ACC.connected else '  Connect account to view orders.'
            t.insert('end', msg + '\n', 'ts')
        else:
            for o in ACC.open_orders[:20]:
                side_tag = 'buy' if o.side == 'BUY' else 'sell'
                fill_pct = float(o.filled) / max(float(o.qty), 1e-9) * 100
                t.insert('end', o.symbol.replace('USDT', '').rjust(8), 'sym')
                t.insert('end', o.side.rjust(5),     side_tag)
                t.insert('end', o.type[:6].rjust(7), 'ts')
                t.insert('end', f'{o.price:,.4f}'.rjust(14), 'price')
                t.insert('end', f'{o.qty:.4f}'.rjust(10), 'ts')
                t.insert('end', f'{fill_pct:.0f}%'.rjust(6), 'ts')
                t.insert('end', o.time.rjust(12) + '\n', 'ts')
        t.config(state='disabled')

    def _filter(self, sym: str) -> None:
        self._acc_filter = sym
        self._update_history()
