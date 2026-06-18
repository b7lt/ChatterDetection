import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from widgets import BasePage
from config import CLASS_COLORS, VISIBLE_CLASSES, pastel
from data_store import DATA
from models import CONTEXT_TAGS


# ── Display metadata ──────────────────────────────────────────────────────────

_TAG_LABEL = {
    "OD":                              "OD Value (in)",
    "Chatter Likelihood":              "Chatter Likelihood (%)",
    "AirRampPressure_Val":             "Air Ramp Pressure (PSI)",
    "FtCounters_AirRampFootage_Total": "Section Footage (ft)",
    "NDC_System_Ovality_Value":        "Ovality (in)",
    "OilHeater_DeliveryTemp_F":        "Oil Delivery Temp (°F)",
    "OilHeater_ReturnTemp_F":          "Oil Return Temp (°F)",
    "PTs_PT_300_Val":                  "PT-300 Pressure (PSI)",
    "PTs_PT_400_Val":                  "PT-400 Pressure (PSI)",
}

_TAG_COLOR = {
    "OD":                              "#2563EB",
    "Chatter Likelihood":              "#F97316",
    "AirRampPressure_Val":             "#7C3AED",
    "FtCounters_AirRampFootage_Total": "#0891B2",
    "NDC_System_Ovality_Value":        "#DB2777",
    "OilHeater_DeliveryTemp_F":        "#EA580C",
    "OilHeater_ReturnTemp_F":          "#D97706",
    "PTs_PT_300_Val":                  "#16A34A",
    "PTs_PT_400_Val":                  "#15803D",
}

_LEFT_OPTIONS  = ["OD"] + list(CONTEXT_TAGS)
_RIGHT_OPTIONS = ["Chatter Likelihood"] + list(CONTEXT_TAGS) + ["None"]


class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Data & Predictions")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # ── Controls ──────────────────────────────────────────────────────────
        controls = ttk.Frame(self)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        self._show_windows = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Show Class Windows",
                        variable=self._show_windows,
                        command=self._redraw).pack(side="left", padx=(0, 12))

        ttk.Separator(controls, orient="vertical").pack(
            side="left", fill="y", padx=(0, 12))

        ttk.Label(controls, text="Left axis:").pack(side="left", padx=(0, 4))
        self._left_var = tk.StringVar(value="OD")
        ttk.Combobox(controls, textvariable=self._left_var,
                     values=_LEFT_OPTIONS, state="readonly",
                     width=30).pack(side="left", padx=(0, 12))
        self._left_var.trace_add("write", lambda *_: self._redraw())

        ttk.Label(controls, text="Right axis:").pack(side="left", padx=(0, 4))
        self._right_var = tk.StringVar(value="Chatter Likelihood")
        ttk.Combobox(controls, textvariable=self._right_var,
                     values=_RIGHT_OPTIONS, state="readonly",
                     width=30).pack(side="left", padx=(0, 12))
        self._right_var.trace_add("write", lambda *_: self._redraw())

        # ── Matplotlib figure ─────────────────────────────────────────────────
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax2 = self.ax1.twinx()

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.mpl_canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(self)
        toolbar_frame.grid(row=3, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.mpl_canvas, toolbar_frame)
        self.toolbar.update()

        self._user_zoomed = False

        def _on_interact(*_):
            try:
                if str(self.toolbar.mode) != "":
                    self._user_zoomed = True
            except Exception:
                pass

        self.mpl_canvas.mpl_connect("button_release_event", _on_interact)
        self.mpl_canvas.mpl_connect("scroll_event",         _on_interact)

        def _home_patched(*_):
            self._user_zoomed = False
            self._redraw()
        self.toolbar.home = _home_patched

        self._show_placeholder()
        self._last_len     = -1
        self._last_classes = -1
        self.after(1000, self._tick)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _get_series(self, var_name: str):
        """
        Return (timestamps, values, y_label, color) for the requested variable.
        timestamps and values are matched-length lists; either may be empty.
        """
        label = _TAG_LABEL.get(var_name, var_name)
        color = _TAG_COLOR.get(var_name, "#6B7280")

        if var_name == "OD":
            return DATA.ts_hist, DATA.od_hist, label, color

        if var_name == "Chatter Likelihood":
            if not DATA.classes:
                return [], [], label, color
            ts   = list(pd.to_datetime(
                [c["start"] for c in DATA.classes], errors="coerce"))
            vals = [c["risk"] * 100.0 for c in DATA.classes]
            return ts, vals, label, color

        if var_name in CONTEXT_TAGS:
            ctx   = DATA.ctx_hist.get(var_name, [])
            n     = min(len(ctx), len(DATA.ts_hist))
            ts    = DATA.ts_hist[:n]
            vals  = ctx[:n]
            # Strip leading NaN padding for cleaner display
            first_ok = next(
                (i for i, v in enumerate(vals)
                 if not (isinstance(v, float) and np.isnan(v))),
                0)
            return ts[first_ok:], vals[first_ok:], label, color

        return [], [], label, color

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _show_placeholder(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.text(0.5, 0.5, "Load data to see history",
                      ha="center", va="center",
                      transform=self.ax1.transAxes,
                      fontsize=12, color="#6B7280")
        self.ax1.set_axis_off()
        self.ax2.set_axis_off()
        self.fig.tight_layout()
        self.mpl_canvas.draw_idle()

    # ── Auto-refresh tick ─────────────────────────────────────────────────────

    def _tick(self):
        cur_len     = len(DATA.od_hist)
        cur_classes = len(DATA.classes) if hasattr(DATA, "classes") else 0
        if cur_len != self._last_len or cur_classes != self._last_classes:
            self._redraw()
            self._last_len     = cur_len
            self._last_classes = cur_classes
        self.after(1000, self._tick)

    # ── Main draw ─────────────────────────────────────────────────────────────

    def _redraw(self):
        # Preserve zoom limits if user has actively panned/zoomed
        if self._user_zoomed:
            saved_xlim  = self.ax1.get_xlim()
            saved_ylim1 = self.ax1.get_ylim()
            saved_ylim2 = self.ax2.get_ylim()

        self.ax1.clear()
        self.ax2.clear()

        left_var  = self._left_var.get()
        right_var = self._right_var.get()

        has_data = bool(DATA.od_hist) or bool(DATA.classes)
        if not has_data:
            self._show_placeholder()
            return

        self.ax1.set_axis_on()
        self.ax2.set_axis_on()

        # ── Left axis ─────────────────────────────────────────────────────────
        l_ts, l_vals, l_label, l_color = self._get_series(left_var)

        if l_vals and l_ts:
            self.ax1.plot(l_ts, l_vals,
                          color=l_color, linewidth=0.6, alpha=0.35)
            k  = max(5, len(l_vals) // 100)
            sm = pd.Series(l_vals).rolling(window=k, min_periods=1).mean().values
            self.ax1.plot(l_ts, sm, color=l_color, linewidth=1.5,
                          label=f"{l_label} (smoothed)")
            self.ax1.set_ylabel(l_label, color=l_color, fontsize=10)
            self.ax1.tick_params(axis="y", labelcolor=l_color)
        else:
            self.ax1.yaxis.set_visible(False)

        # ── Right axis ────────────────────────────────────────────────────────
        r_ts, r_vals, r_label, r_color = [], [], "", "#6B7280"

        if right_var != "None":
            r_ts, r_vals, r_label, r_color = self._get_series(right_var)
            if r_vals and r_ts:
                self.ax2.plot(r_ts, r_vals, color=r_color, linewidth=1.5,
                              label=r_label)
                self.ax2.set_ylabel(r_label, color=r_color, fontsize=10)
                self.ax2.tick_params(axis="y", labelcolor=r_color)
                if right_var == "Chatter Likelihood":
                    self.ax2.set_ylim([0, 110])
            else:
                self.ax2.yaxis.set_visible(False)
        else:
            self.ax2.yaxis.set_visible(False)

        # ── Shaded class-window bands ─────────────────────────────────────────
        if self._show_windows.get() and DATA.classes:
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
                    self.ax1.axvspan(
                        pd.Timestamp(m["start"]), pd.Timestamp(m["end"]),
                        facecolor=pastel(color, 0.3),
                        alpha=0.4, linewidth=0, zorder=0)
                except Exception:
                    pass

        # ── Legend ────────────────────────────────────────────────────────────
        handles, labels = [], []
        if l_vals:
            handles.append(Line2D([0], [0], color=l_color, linewidth=1.5))
            labels.append(l_label)
        if right_var != "None" and r_vals:
            handles.append(Line2D([0], [0], color=r_color, linewidth=1.5))
            labels.append(r_label)
        if self._show_windows.get() and DATA.classes:
            for name, color in CLASS_COLORS.items():
                if name in VISIBLE_CLASSES:
                    handles.append(Patch(facecolor=pastel(color, 0.3), edgecolor="none"))
                    labels.append(name)
        if handles:
            self.ax1.legend(handles, labels, loc="upper left", fontsize=8)

        self.ax1.set_xlabel("Time", fontsize=10)
        self.fig.autofmt_xdate(rotation=30, ha="right")
        self.fig.tight_layout()

        if self._user_zoomed:
            self.ax1.set_xlim(saved_xlim)
            self.ax1.set_ylim(saved_ylim1)
            self.ax2.set_ylim(saved_ylim2)

        self.mpl_canvas.draw_idle()