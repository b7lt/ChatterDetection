import math
from datetime import datetime
from tkinter import ttk

import numpy as np
import pandas as pd

from widgets import BasePage, Gauge
from pages.live import LiveTimeSeries
from data_store import DATA


class ResultsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Results")

        grid = ttk.Frame(self); grid.grid(row=1, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(1, weight=1)
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        left = ttk.Frame(grid); left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="OD vs Time (Live)", style="Subhead.TLabel").pack(anchor="w")
        self.live = LiveTimeSeries(left)
        self.live.pack(fill="both", expand=True, pady=(6, 0))

        right_top = ttk.Frame(grid); right_top.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right_top, text="OK / NG Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right_top, width=360, height=200)
        self.gauge.pack(fill="both", expand=True, pady=(6, 0))

        self.pred_label = ttk.Label(right_top, text="Predicted class: —", style="KPI.TLabel")
        self.pred_label.pack(anchor="w", pady=(6, 0))
        self.pred_conf = ttk.Label(right_top, text="Confidence: —", style="KPI.TLabel")
        self.pred_conf.pack(anchor="w")

        self.fft_metric = ttk.Label(right_top, text="FFT: —", style="KPI.TLabel")
        self.fft_metric.pack(anchor="w", pady=(2, 0))

        self.after(1000, self._tick)

    def _tick(self):
        pct = 50

        if DATA.od and DATA.classes:
            lbl, class_risk = DATA.current_class()
            if class_risk is not None:
                pct = class_risk * 100.0
                self.pred_label.config(text=f"Predicted class: {lbl}")
                self.pred_conf.config(text=f"Confidence: {pct:0.1f}%")
            else:
                pct = 0
                self.pred_label.config(text="Error")
                self.pred_conf.config(text="Error")
        elif DATA.od:
            # data loaded but no model selected yet
            pct = 50 + 15 * math.sin(datetime.now().timestamp() / 2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please select a model")
        else:
            # no data — demo mode
            pct = 50 + 15 * math.sin(datetime.now().timestamp() / 2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please import data and select a model")

        self.gauge.set_value(pct)

        fft_info = self._compute_fft_metric(n=24000)

        if fft_info is None:
            self.fft_metric.config(text="FFT: inconclusive (not enough data or flat signal)")
        else:
            self.fft_metric.config(
                text=(
                    f"FFT: peak power {fft_info['peak_power']:.3g} "
                    f"at {fft_info['peak_freq']:.3g} Hz "
                    f"(prominence {fft_info['prominence'] * 100:.1f}%)"
                )
            )

        self.after(1000, self._tick)

    def _compute_fft_metric(self, n=512):
        """
        Look at the most recent n OD samples and compute:
          - peak frequency (Hz)
          - peak power (arbitrary units)
          - prominence = peak_power / total_power in non-DC bins
        Returns None if there isn't enough data or signal is degenerate.
        """
        if not DATA.od:
            return None

        y = np.asarray(DATA.recent_window(n), dtype=float)
        if y.size < 16:
            return None

        y = y - np.mean(y)
        if not np.any(np.isfinite(y)) or np.allclose(y, 0.0, atol=1e-12):
            return None

        # estimate sampling period from timestamps (fallback to 1 Hz)
        fs = 1.0
        try:
            if DATA.ts_dt and len(DATA.ts_dt) >= len(y):
                t = pd.to_datetime(DATA.ts_dt[-len(y):], errors="coerce")
                t = t[~pd.isna(t)]
                if len(t) >= 2:
                    dt_sec = np.median(
                        np.diff(t).astype("timedelta64[ns]").astype(np.float64)
                    ) / 1e9
                    if dt_sec > 0:
                        fs = 1.0 / dt_sec
        except Exception:
            pass

        nfft = len(y)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        fft_vals = np.fft.rfft(y)
        psd = (np.abs(fft_vals) ** 2) / nfft

        if psd.size <= 1:
            return None

        # ignore DC component at index 0
        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]

        peak_idx = int(np.argmax(psd_no_dc))
        peak_freq = float(freqs_no_dc[peak_idx])
        peak_power = float(psd_no_dc[peak_idx])
        total_power = float(np.sum(psd_no_dc))

        if not np.isfinite(peak_power) or total_power <= 0 or not np.isfinite(total_power):
            return None

        prominence = float(peak_power / total_power)

        # heuristic: require a reasonably dominant peak
        periodic = prominence > 0.1

        return {
            "fs": fs,
            "peak_freq": peak_freq,
            "peak_power": peak_power,
            "prominence": prominence,
            "periodic": periodic,
        }
