import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_store import DATA


class CorrelationWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("OD ↔ Secondary Correlation")
        self.minsize(980, 720)

        if DATA.paired_df is None or DATA.paired_df.empty:
            ttk.Label(self, text="Load a secondary file first.", padding=12).pack()
            return

        # Top stats / controls
        top = ttk.Frame(self, padding=12)
        top.pack(side="top", fill="x")

        stats = DATA.corr_stats()
        rtxt = f"r = {stats['pearson_r']:.3f}  |  best lag: {stats['best_lag']} samples (r={stats['r_at_best_lag']:.3f})"
        sign = "POSITIVE" if stats["pearson_r"] >= 0 else "NEGATIVE"
        color = "#16A34A" if stats["pearson_r"] >= 0 else "#DC2626"

        ttk.Label(top, text=f"Paired rows: {stats['n']}", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 20))
        ttk.Label(top, text=rtxt).grid(row=0, column=1, sticky="w", padx=(0, 20))
        badge = ttk.Label(top, text=f"Tendency: {sign}", foreground=color, font=("Segoe UI", 10, "bold"))
        badge.grid(row=0, column=2, sticky="w")

        ctrl = ttk.Frame(top)
        ctrl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(ctrl, text="Max lag (samples):").grid(row=0, column=0, sticky="w")
        self.maxlag_var = tk.IntVar(value=300)
        ttk.Entry(ctrl, textvariable=self.maxlag_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 16))
        ttk.Label(ctrl, text="Rolling window (samples):").grid(row=0, column=2, sticky="w")
        self.win_var = tk.IntVar(value=200)
        ttk.Entry(ctrl, textvariable=self.win_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 16))
        ttk.Button(ctrl, text="Update plots", command=self._refresh_all).grid(row=0, column=4)

        # Tabs
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True, padx=12, pady=12)

        self.tab_overlay = ttk.Frame(nb); nb.add(self.tab_overlay, text="Time Overlay")
        self.tab_scatter = ttk.Frame(nb); nb.add(self.tab_scatter, text="Scatter & Density")
        self.tab_lag     = ttk.Frame(nb); nb.add(self.tab_lag,     text="Corr vs Lag")
        self.tab_roll    = ttk.Frame(nb); nb.add(self.tab_roll,    text="Rolling Corr")

        # Matplotlib canvases
        self.fig_overlay = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_overlay  = self.fig_overlay.add_subplot(111)
        self.cv_overlay  = FigureCanvasTkAgg(self.fig_overlay, master=self.tab_overlay)
        self.cv_overlay.get_tk_widget().pack(fill="both", expand=True)

        self.fig_scatter = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_scatter  = self.fig_scatter.add_subplot(111)
        self.cv_scatter  = FigureCanvasTkAgg(self.fig_scatter, master=self.tab_scatter)
        self.cv_scatter.get_tk_widget().pack(fill="both", expand=True)

        self.fig_lag = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_lag  = self.fig_lag.add_subplot(111)
        self.cv_lag  = FigureCanvasTkAgg(self.fig_lag, master=self.tab_lag)
        self.cv_lag.get_tk_widget().pack(fill="both", expand=True)

        self.fig_roll = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_roll  = self.fig_roll.add_subplot(111)
        self.cv_roll  = FigureCanvasTkAgg(self.fig_roll, master=self.tab_roll)
        self.cv_roll.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_all()

    def _refresh_all(self):
        self._draw_overlay()
        self._draw_scatter()
        self._draw_lag()
        self._draw_rolling()

    def _draw_overlay(self):
        df = DATA.paired_df.copy()
        df["od_z"]  = (df["od"]  - df["od"].mean())  / (df["od"].std(ddof=0)  + 1e-12)
        df["sec_z"] = (df["sec"] - df["sec"].mean()) / (df["sec"].std(ddof=0) + 1e-12)

        self.ax_overlay.clear()
        t = pd.to_datetime(df["t"], errors="coerce")
        self.ax_overlay.plot(t, df["od_z"],  linewidth=1.5, label="OD (z-score)")
        self.ax_overlay.plot(t, df["sec_z"], linewidth=1.5, label="Secondary (z-score)")
        self.ax_overlay.axhline(0, linewidth=0.8, color="#999999")
        self.ax_overlay.set_title("Time Overlay (z-scored)")
        self.ax_overlay.set_xlabel("time"); self.ax_overlay.set_ylabel("z-score")
        self.ax_overlay.legend(loc="upper right")
        self.fig_overlay.tight_layout()
        self.cv_overlay.draw()

    def _draw_scatter(self):
        df = DATA.paired_df
        x = df["od"].to_numpy(dtype=float)
        y = df["sec"].to_numpy(dtype=float)

        self.ax_scatter.clear()
        hb = self.ax_scatter.hexbin(x, y, gridsize=40, bins="log")
        self.fig_scatter.colorbar(hb, ax=self.ax_scatter, fraction=0.046, pad=0.04, label="log density")

        if len(x) >= 2:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 200)
            self.ax_scatter.plot(xx, a * xx + b, linewidth=2, label=f"Fit: y={a:.3f}x+{b:.3f}")

        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
        self.ax_scatter.set_title(f"Scatter & Density  (r={r:.3f})")
        self.ax_scatter.set_xlabel("OD"); self.ax_scatter.set_ylabel("Secondary")
        self.ax_scatter.legend(loc="best")
        self.fig_scatter.tight_layout()
        self.cv_scatter.draw()

    def _draw_lag(self):
        maxlag = max(5, int(self.maxlag_var.get()))
        lags, r = DATA.lag_corr_curve(max_lag_samples=maxlag)
        self.ax_lag.clear()
        if lags.size:
            self.ax_lag.plot(lags, r, linewidth=2)
            self.ax_lag.axhline(0, linewidth=0.8, color="#999999")
            k = int(lags[np.nanargmax(r)])
            rmax = float(np.nanmax(r))
            self.ax_lag.axvline(k, linestyle="--", linewidth=1.2)
            self.ax_lag.set_title(f"Correlation vs Lag (best: {k} samples, r={rmax:.3f})")
            self.ax_lag.set_xlabel("lag (samples, + = OD leads)"); self.ax_lag.set_ylabel("Pearson r")
        else:
            self.ax_lag.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_lag.axis("off")
        self.fig_lag.tight_layout()
        self.cv_lag.draw()

    def _draw_rolling(self):
        win = max(20, int(self.win_var.get()))
        t, r = DATA.rolling_corr(win_samples=win, step=max(5, win // 10))
        self.ax_roll.clear()
        if r.size:
            self.ax_roll.plot(t, r, linewidth=2)
            self.ax_roll.axhline(0, linewidth=0.8, color="#999999")
            self.ax_roll.set_title(f"Rolling Correlation (window={win} samples)")
            self.ax_roll.set_xlabel("time"); self.ax_roll.set_ylabel("Pearson r")
        else:
            self.ax_roll.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_roll.axis("off")
        self.fig_roll.tight_layout()
        self.cv_roll.draw()
