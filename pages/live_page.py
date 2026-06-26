import math
from datetime import datetime

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from widgets import BasePage, Gauge
from pages.live import LiveTimeSeries
from data_store import DATA
from models import CONTEXT_TAGS
from config import FOOTAGE_TAG, OD_TAG, PRESSURE_TAG


# ── Display metadata ──────────────────────────────────────────────────────────

_TAG_SHORT = {
    "AirRampPressure_Val":             "Air Ramp PSI",
    "FtCounters_AirRampFootage_Total": "Section Footage",
    "NDC_System_Ovality_Value":        "Ovality",
    "OilHeater_DeliveryTemp_F":        "Oil Deliv. Temp",
    "OilHeater_ReturnTemp_F":          "Oil Ret. Temp",
    "PTs_PT_300_Val":                  "PT-300",
    "PTs_PT_400_Val":                  "PT-400",
}

_TAG_UNITS = {
    "AirRampPressure_Val":             "PSI",
    "FtCounters_AirRampFootage_Total": "ft",
    "NDC_System_Ovality_Value":        "in",
    "OilHeater_DeliveryTemp_F":        "°F",
    "OilHeater_ReturnTemp_F":          "°F",
    "PTs_PT_300_Val":                  "PSI",
    "PTs_PT_400_Val":                  "PSI",
}

# Segment KPI strip: (dict_key, display_label, format_spec, snapshot_key, unit)
_KPI_DEFS = [
    ("segment",  "Segment",           "s",   "current_segment_id",              ""),
    ("state",    "State",             "s",   "segment_state",                   ""),
    ("footage",  "Section Footage",   ".0f", FOOTAGE_TAG,                      "ft"),
    ("pressure", "Air Ramp Pressure", ".3f", PRESSURE_TAG,                     "PSI"),
    ("od",       "Current OD",        ".5f", OD_TAG,                           "in"),
    ("ovality",  "Ovality",           ".5f", "NDC_System_Ovality_Value",        "in"),
    ("oil_temp", "Oil Deliv. Temp",   ".1f", "OilHeater_DeliveryTemp_F",        "°F"),
    ("pt300",    "PT-300",            ".2f", "PTs_PT_300_Val",                  "PSI"),
    ("pt400",    "PT-400",            ".2f", "PTs_PT_400_Val",                  "PSI"),
]


class LivePage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Live Data & Predictions")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)   # main content
        self.rowconfigure(2, weight=0)   # segment info strip

        # ── Row 1: main content ───────────────────────────────────────────────
        main = ttk.Frame(self)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=1, minsize=270)
        main.rowconfigure(0, weight=1)

        # Left: Live OD time-series
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="Live OD", style="Subhead.TLabel").pack(anchor="w")
        self.live = LiveTimeSeries(left)
        self.live.pack(fill="both", expand=True, pady=(6, 0))

        # Right: Gauge stacked above explainability
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)   # explainability LabelFrame expands

        ttk.Label(right, text="Chatter Indicator",
                  style="Subhead.TLabel").grid(row=0, column=0, sticky="w")
        self.gauge = Gauge(right)
        self.gauge.grid(row=1, column=0, sticky="ew", pady=(4, 4))

        # Explainability panel
        expl_outer = ttk.LabelFrame(right, text="Context Feature Attribution",
                                    padding=(6, 4))
        expl_outer.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        expl_outer.columnconfigure(0, weight=1)
        expl_outer.rowconfigure(0, weight=1)

        self.expl_fig    = Figure(figsize=(3.2, 3.0), dpi=90)
        self.expl_ax     = self.expl_fig.add_subplot(111)
        self.expl_canvas = FigureCanvasTkAgg(self.expl_fig, master=expl_outer)
        self.expl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Operator suggestions below the chart
        self.suggest_var = tk.StringVar(value="")
        ttk.Label(expl_outer, textvariable=self.suggest_var,
                  foreground="#6B7280", wraplength=240, justify="left",
                  font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w",
                                             pady=(4, 0))

        self._draw_expl_placeholder()

        # ── Row 2: segment info strip ─────────────────────────────────────────
        seg_frame = ttk.LabelFrame(self, text="Current Segment", padding=(8, 4))
        seg_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self._kpi_vars: dict[str, tk.StringVar] = {}
        for col, (key, label, fmt, src_key, unit) in enumerate(_KPI_DEFS):
            seg_frame.columnconfigure(col, weight=1)
            kf = ttk.Frame(seg_frame)
            kf.grid(row=0, column=col, padx=6, sticky="ew")
            ttk.Label(kf, text=label, font=("Segoe UI", 8),
                      foreground="#6B7280").pack(anchor="n")
            var = tk.StringVar(value="—")
            self._kpi_vars[key] = var
            ttk.Label(kf, textvariable=var,
                      font=("Segoe UI", 11, "bold")).pack(anchor="n")
            ttk.Label(kf, text=unit, font=("Segoe UI", 8),
                      foreground="#9CA3AF").pack(anchor="n")

        self._last_class_count = -1
        self.after(1000, self._tick)

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _draw_expl_placeholder(self, msg="Load a HybridChatterNet\nmodel to see attribution"):
        self.expl_ax.clear()
        self.expl_ax.text(0.5, 0.5, msg, ha="center", va="center",
                          fontsize=9, color="#6B7280",
                          transform=self.expl_ax.transAxes)
        self.expl_ax.axis("off")
        self.expl_fig.tight_layout()
        self.expl_canvas.draw_idle()

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _tick(self):
        # Gauge
        if DATA.od and DATA.classes:
            _, risk = DATA.current_class()
            pct = (risk * 100.0) if risk is not None else 0
        else:
            pct = 50 + 15 * math.sin(datetime.now().timestamp() / 2.0)
        self.gauge.set_value(pct)

        # Segment KPIs
        self._update_kpis()

        # Explainability — only recompute when a new window is classified
        cc = len(DATA.classes)
        if cc != self._last_class_count:
            self._update_explainability()
            self._last_class_count = cc

        self.after(1000, self._tick)

    # ── Segment KPI update ────────────────────────────────────────────────────

    def _update_kpis(self):
        snap = DATA.live_snapshot

        def _val(src_key, fmt):
            # Live snapshot has priority; fall back to context_data last value
            if snap and src_key in snap:
                v = snap[src_key]
            elif src_key == "NDC_System_OD_Value":
                v = DATA.od[-1] if DATA.od else None
            else:
                d = DATA.context_data.get(src_key, [])
                v = d[-1] if d else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            if fmt == "s":
                return str(v).title() if src_key == "segment_state" else str(v)
            try:
                return format(float(v), fmt)
            except (ValueError, TypeError):
                return "—"

        for key, label, fmt, src_key, unit in _KPI_DEFS:
            self._kpi_vars[key].set(_val(src_key, fmt))

    # ── Explainability update ─────────────────────────────────────────────────

    def _update_explainability(self):
        result = DATA.explain_last_window()

        if result is None:
            if DATA.model is not None and not DATA._is_hybrid_model():
                self._draw_expl_placeholder("Explainability requires\nHybridChatterNet")
            elif not DATA.classes:
                self._draw_expl_placeholder("Awaiting first\nclassification…")
            else:
                self._draw_expl_placeholder("Attribution\nnot available")
            self.suggest_var.set("")
            return

        tags         = result["tags"]
        attributions = np.asarray(result["attributions"], dtype=float)
        grads        = np.asarray(result["grads"],        dtype=float)
        raw_vals     = result["raw_values"]
        prob         = result["chatter_prob"]

        max_abs = max(float(np.abs(attributions).max()), 1e-8)
        norm    = attributions / max_abs

        labels = [_TAG_SHORT.get(t, t) for t in tags]
        colors = ["#DC2626" if v > 0 else "#2563EB" for v in norm]

        self.expl_ax.clear()
        self.expl_ax.barh(labels, norm, color=colors, alpha=0.85, height=0.55)
        self.expl_ax.axvline(0, color="#374151", linewidth=0.8)
        self.expl_ax.set_xlim(-1.25, 1.25)
        self.expl_ax.set_xlabel("← safer  |  riskier →", fontsize=7)

        # risk_color = ("#DC2626" if prob > 0.55 else
        #               "#F59E0B" if prob > 0.35 else "#16A34A")
        # self.expl_ax.set_title(f"Chatter prob: {prob*100:.0f}%",
        #                        fontsize=9, fontweight="bold", color=risk_color)
        self.expl_ax.tick_params(axis="y", labelsize=7)
        self.expl_ax.tick_params(axis="x", labelsize=7)
        self.expl_ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        self.expl_ax.set_axis_on()
        self.expl_fig.tight_layout(pad=0.5)
        self.expl_canvas.draw_idle()

        # ── Operator suggestions ──────────────────────────────────────────────
        if prob > 0.35:
            # Sort by attribution descending — most chatter-driving feature first
            ranked = sorted(
                zip(attributions.tolist(), grads.tolist(), tags, raw_vals),
                key=lambda x: -x[0]
            )
            lines = []
            for attr, grad, tag, val in ranked:
                if attr < 0.08 * max_abs:
                    break
                short = _TAG_SHORT.get(tag, tag)
                unit  = _TAG_UNITS.get(tag, "")
                arrow = "↓ Reduce" if grad > 0 else "↑ Increase"
                try:
                    val_str = f"{float(val):.3g}"
                except (ValueError, TypeError):
                    val_str = str(val)
                lines.append(f"{arrow} {short}  ({val_str} {unit})")
                if len(lines) >= 3:
                    break
            self.suggest_var.set(
                ("To reduce risk:\n" + "\n".join(lines)) if lines else "")
        else:
            self.suggest_var.set("✓ Low chatter risk")
