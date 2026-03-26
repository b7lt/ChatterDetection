import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from widgets import BasePage
from config import CLASS_COLORS, VISIBLE_CLASSES, pastel
from data_store import DATA


class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Data & Predictions")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Toggle controls
        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        self._show_od   = tk.BooleanVar(value=True)
        self._show_conf = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Show OD",
                        variable=self._show_od,   command=self._redraw).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(controls, text="Show Chatter Likelihood",
                        variable=self._show_conf, command=self._redraw).pack(side="left")

        # Matplotlib figure with twin y-axes
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(111)   # left axis  — OD
        self.ax2 = self.ax1.twinx()            # right axis — likelihood

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.mpl_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=3, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.mpl_canvas, toolbar_frame)
        self.toolbar.update()

        # Zoom state: set True on mouse/scroll interaction, False on Home.
        # Avoids relying on matplotlib private nav-stack internals.
        self._user_zoomed = False

        def _on_interact(*_):
            # Any mouse release or scroll while a toolbar tool is active = user zoomed
            try:
                if str(self.toolbar.mode) != '':
                    self._user_zoomed = True
            except Exception:
                pass

        self.mpl_canvas.mpl_connect('button_release_event', _on_interact)
        self.mpl_canvas.mpl_connect('scroll_event',         _on_interact)

        # Home: fresh auto-scaled redraw of all current data
        def _home_patched(*_):
            self._user_zoomed = False
            self._redraw()
        self.toolbar.home = _home_patched

        self._show_placeholder()

        self._last_len     = -1
        self._last_classes = -1
        self.after(1000, self._tick)

    # ------------------------------------------------------------------
    def _show_placeholder(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.text(0.5, 0.5, "Load data to see history",
                      ha="center", va="center", transform=self.ax1.transAxes,
                      fontsize=12, color="#6B7280")
        self.ax1.set_axis_off()
        self.ax2.set_axis_off()
        self.fig.tight_layout()
        self.mpl_canvas.draw_idle()

    def _tick(self):
        current_len     = len(DATA.od_hist)
        current_classes = len(DATA.classes) if hasattr(DATA, 'classes') else 0
        if current_len != self._last_len or current_classes != self._last_classes:
            self._redraw()
            self._last_len     = current_len
            self._last_classes = current_classes
        self.after(1000, self._tick)

    # ------------------------------------------------------------------
    def _redraw(self):
        # Save limits only when the user has actively zoomed/panned
        if self._user_zoomed:
            _xlim  = self.ax1.get_xlim()
            _ylim1 = self.ax1.get_ylim()
            _ylim2 = self.ax2.get_ylim()

        self.ax1.clear()
        self.ax2.clear()

        show_od   = self._show_od.get()   and bool(DATA.od_hist)
        show_conf = self._show_conf.get() and bool(DATA.classes)

        if not DATA.od_hist and not DATA.classes:
            self._show_placeholder()
            return

        self.ax1.set_axis_on()
        self.ax2.set_axis_on()

        # ---- OD line (left axis) ------------------------------------
        if show_od and DATA.ts_hist:
            ts = DATA.ts_hist
            od = DATA.od_hist
            self.ax1.plot(ts, od, color="#BFDBFE", linewidth=0.6, alpha=0.7)
            k  = max(5, len(od) // 100)
            sm = pd.Series(od).rolling(window=k, min_periods=1).mean().values
            self.ax1.plot(ts, sm, color="#2563EB", linewidth=1.5, label="OD (smoothed)")
            self.ax1.set_ylabel("OD Value", color="#2563EB", fontsize=10)
            self.ax1.tick_params(axis='y', labelcolor="#2563EB")
        else:
            self.ax1.yaxis.set_visible(False)

        # ---- Likelihood line + shaded bands (right axis) ------------
        if show_conf and DATA.classes:
            window_times = pd.to_datetime([c["start"] for c in DATA.classes], errors="coerce")
            confidences  = [c["risk"] * 100.0 for c in DATA.classes]
            self.ax2.plot(window_times, confidences,
                          color="#F97316", linewidth=1.5, label="Chatter Likelihood")
            self.ax2.set_ylabel("Chatter Likelihood (%)", color="#F97316", fontsize=10)
            self.ax2.tick_params(axis='y', labelcolor="#F97316")
            self.ax2.set_ylim([0, 110])

            # Merge consecutive same-label spans → fewer axvspan calls
            merged = []
            for span in DATA.classes:
                lbl = span.get("label", "")
                if lbl not in VISIBLE_CLASSES:
                    continue
                if merged and merged[-1]["label"] == lbl:
                    merged[-1]["end"] = span["end"]
                else:
                    merged.append({"label": lbl,
                                   "start": span["start"],
                                   "end":   span["end"]})
            for m in merged:
                color = CLASS_COLORS.get(m["label"], "#BBBBBB")
                try:
                    t0 = pd.Timestamp(m["start"])
                    t1 = pd.Timestamp(m["end"])
                except Exception:
                    continue
                self.ax1.axvspan(t0, t1,
                                 facecolor=pastel(color, 0.3),
                                 alpha=0.4, linewidth=0, zorder=0)
        else:
            self.ax2.yaxis.set_visible(False)

        # ---- Unified legend -----------------------------------------
        handles, labels = [], []
        if show_od:
            handles.append(Line2D([0], [0], color="#2563EB", linewidth=1.5))
            labels.append("OD (smoothed)")
        if show_conf:
            handles.append(Line2D([0], [0], color="#F97316", linewidth=1.5))
            labels.append("Chatter Likelihood (%)")
        for name, color in CLASS_COLORS.items():
            if name in VISIBLE_CLASSES:
                handles.append(Patch(facecolor=pastel(color, 0.3), edgecolor="none"))
                labels.append(name)
        if handles:
            self.ax1.legend(handles, labels, loc="upper left", fontsize=8)

        self.ax1.set_xlabel("Time", fontsize=10)
        self.fig.autofmt_xdate(rotation=30, ha="right")
        self.fig.tight_layout()

        # Restore zoom if user had actively adjusted the view
        if self._user_zoomed:
            self.ax1.set_xlim(_xlim)
            self.ax1.set_ylim(_ylim1)
            self.ax2.set_ylim(_ylim2)

        self.mpl_canvas.draw_idle()
