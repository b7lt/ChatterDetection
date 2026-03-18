from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from widgets import BasePage
from config import CLASS_COLORS, VISIBLE_CLASSES, pastel
from data_store import DATA
from status_bar import status


class AnalysisPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Modeling & Analysis")

        top = ttk.Frame(self); top.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.status_lbl = ttk.Label(top, text="", foreground="#6B7280")
        self.status_lbl.grid(row=0, column=0, sticky="w", padx=12)

        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1); self.columnconfigure(0, weight=1)

        # Confidence timeline of model output
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._overlay_images = []

        self.ax.text(0.5, 0.5, "Load data and select a model to see predictions",
                     ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

        # FFT figure: latest time window FFT
        self.fft_fig = Figure(figsize=(10, 3), dpi=100)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=area)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        self.fft_ax.text(0.5, 0.5, "FFT of latest window will appear here",
                         ha="center", va="center", fontsize=11, color="#6B7280")
        self.fft_ax.set_axis_off()
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()

        self.after(1500, self._tick_fft)

    def _tick_fft(self):
        self.update_fft_plot(n=24000)
        self.after(1500, self._tick_fft)

    def update_confidence_timeline(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return

        if DATA.model is None:
            messagebox.showinfo("No Model", "Please select a model first.")
            return

        status("Computing confidence timeline.")

        self._overlay_images.clear()
        self.ax.clear()

        ws = DATA.window_size
        num_windows = len(DATA.od) // ws

        if num_windows < 1:
            self.ax.text(0.5, 0.5, "Not enough data for the selected window size",
                         ha="center", va="center", fontsize=12)
            self.ax.axis("off")
            self.canvas.draw()
            return

        confidences = []
        window_times = []

        if not DATA.ts_dt:
            DATA.ts_dt = pd.to_datetime(DATA.ts, errors='coerce').tolist()
            if not DATA.ts_dt or all(pd.isna(t) for t in DATA.ts_dt):
                DATA.ts_dt = [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(DATA.od))]

        for i in range(num_windows):
            start_idx = i * ws
            end_idx = start_idx + ws
            if end_idx > len(DATA.od):
                break

            window = DATA.od[start_idx:end_idx]
            features_dict = DATA.extract_features(window)
            X = pd.DataFrame([features_dict])
            probas = DATA.model.predict_proba(X)
            confidences.append(probas[0, 1] * 100)

            mid_idx = start_idx + ws // 2
            if mid_idx < len(DATA.ts_dt):
                window_times.append(DATA.ts_dt[mid_idx])
            else:
                window_times.append(DATA.ts_dt[-1])

        if not confidences:
            self.ax.text(0.5, 0.5, "Not enough data to compute timeline",
                         ha="center", va="center", fontsize=12)
            self.ax.axis("off")
            self.canvas.draw()
            return

        self.ax.plot(window_times, confidences, label="Chatter likelihood", color="#2563EB", linewidth=2)

        # shaded class bands
        for span in getattr(DATA, "classes", []):
            label = span.get("label", "UNCERTAIN")
            color = CLASS_COLORS.get(label, "#BBBBBB")
            start_time = span.get("start")
            end_time = span.get("end")
            if start_time is None or end_time is None:
                continue
            self.ax.axvspan(start_time, end_time,
                            facecolor=pastel(color, 0.25),
                            alpha=0.25,
                            linewidth=0, zorder=0)

        self.ax.set_xlabel('Time', fontsize=11)
        self.ax.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax.set_title(f'Chatter Confidence in each Window over Time (Window Size: {ws} samples)',
                          fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, zorder=0)
        self.ax.set_ylim([0, 105])

        handles, labels = self.ax.get_legend_handles_labels()
        from matplotlib.patches import Patch
        for name, color in CLASS_COLORS.items():
            if name in VISIBLE_CLASSES:
                handles.append(Patch(facecolor=pastel(color, 0.25), label=name))
                labels.append(name)
        self.ax.legend(handles, labels, loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        self.status_lbl.config(text=f"Timeline updated: {len(confidences)} windows analyzed")
        status(f"Confidence timeline computed for {len(confidences)} windows")

    def update_fft_plot(self, n=512):
        """
        Plot FFT of the latest n OD samples in the bottom chart.
        Shows 'inconclusive' style messages if there is not enough data or no clear peak.
        """
        self.fft_ax.clear()

        if not DATA.od:
            self.fft_ax.text(0.5, 0.5, "No data loaded", ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        y = np.asarray(DATA.recent_window(n), dtype=float)
        if y.size < 16:
            self.fft_ax.text(0.5, 0.5, "Not enough samples for FFT", ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        y = y - np.mean(y)
        if not np.any(np.isfinite(y)) or np.allclose(y, 0.0, atol=1e-12):
            self.fft_ax.text(0.5, 0.5, "FFT inconclusive (flat signal)", ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        # Estimate sampling rate from OD timestamps (fallback to 1 Hz)
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
            self.fft_ax.text(0.5, 0.5, "FFT inconclusive", ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        # Ignore DC bin when plotting and measuring
        freqs_plot = freqs[1:]
        psd_plot = psd[1:]

        self.fft_ax.plot(freqs_plot, psd_plot)
        self.fft_ax.set_xlabel("Frequency (Hz)")
        self.fft_ax.set_ylabel("Power")
        self.fft_ax.set_title("FFT of Latest Window")
        self.fft_ax.grid(True, alpha=0.3)
        self.fft_ax.set_axis_on()

        # Highlight dominant frequency if clearly periodic
        peak_idx = int(np.argmax(psd_plot))
        peak_freq = float(freqs_plot[peak_idx])
        peak_power = float(psd_plot[peak_idx])
        total_power = float(np.sum(psd_plot))
        prominence = peak_power / total_power if total_power > 0 else 0.0

        if total_power > 0 and prominence > 0.30:
            self.fft_ax.axvline(peak_freq, linestyle="--", alpha=0.7)
            self.fft_ax.text(
                peak_freq, peak_power,
                f"{peak_freq:.3g} Hz",
                rotation=90, va="bottom", ha="right", fontsize=9
            )

        self.fft_fig.tight_layout()
        self.fft_canvas.draw()
