"""
UI widgets
"""
import math
import tkinter as tk
from tkinter import ttk


class Gauge(ttk.Frame):
    # Zones match thresholds in DataStore.get_label_from_risk_prob:
    #   No Chatter < 40%, Mild Chatter 40-70%, Heavy Chatter >= 70%
    _ZONES = [
        (0,  40,  "#16A34A", "No\nChatter"),
        (40, 70,  "#D97706", "Mild\nChatter"),
        (70, 100, "#DC2626", "Heavy\nChatter"),
    ]
    _PAD = 16  # pixels of padding around the arc inside the canvas

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0, height=240)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._needle = None
        self._status = ttk.Label(self, text="—", font=("Segoe UI", 13, "bold"))
        self._status.grid(row=1, column=0, pady=(6, 0))
        self._cx = self._cy = self._r = None
        self._pct = 0.0
        self._resize_job = None
        # Bind to the canvas so event.width/height are always the actual canvas size
        self.canvas.bind("<Configure>", self._on_canvas_resize)

    def _on_canvas_resize(self, event):
        # Debounce: only redraw 50 ms after the last resize event to avoid lag
        if self._resize_job:
            self.after_cancel(self._resize_job)
        w, h = event.width, event.height
        self._resize_job = self.after(50, self._redraw, w, h)

    def _redraw(self, w, h):
        self._resize_job = None
        if w < 20 or h < 20:
            return

        pad = self._PAD
        # r is the largest radius that fits with padding on all sides.
        # Semicircle occupies width=2r, height=r, so:
        #   r <= w/2 - pad  (left/right padding)
        #   r <= h - 2*pad  (top + some bottom clearance)
        r = max(10, min(w // 2 - pad, h - 2 * pad))
        cx = w // 2
        # Center the semicircle vertically: bbox is [cy-r, cy], so center at cy-r/2.
        # Setting cy-r/2 = h/2  →  cy = h/2 + r/2
        cy = h // 2 + r // 2

        self._cx, self._cy, self._r = cx, cy, r
        self.canvas.delete("all")
        self._needle = None

        # Coloured arcs — tkinter: start=0 at 3-o'clock, positive extent = CCW
        # We map pct=0 → 180° (left) and pct=100 → 0° (right)
        for lo, hi, color, _ in self._ZONES:
            start_ang = 180 - lo * 180 / 100
            extent = -(hi - lo) * 180 / 100
            self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r,
                                   start=start_ang, extent=extent,
                                   fill=color, outline="")

        # Inner white circle → ring appearance
        inner = int(r * 0.55)
        self.canvas.create_oval(cx - inner, cy - inner, cx + inner, cy + inner,
                                fill="white", outline="white")

        # Zone labels at arc midpoints
        for lo, hi, _, label in self._ZONES:
            mid_pct = (lo + hi) / 2
            ang = math.radians(180 - mid_pct * 180 / 100)
            lr = int(r * 0.78)
            lx = cx + lr * math.cos(ang)
            ly = cy - lr * math.sin(ang)
            self.canvas.create_text(lx, ly, text=label, fill="white",
                                    font=("Segoe UI", 8, "bold"), justify="center")

        # Tick marks at zone boundaries
        for pct in (0, 40, 70, 100):
            ang = math.radians(180 - pct * 180 / 100)
            x0 = cx + (r - 14) * math.cos(ang)
            y0 = cy - (r - 14) * math.sin(ang)
            x1 = cx + (r + 2) * math.cos(ang)
            y1 = cy - (r + 2) * math.sin(ang)
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill="#334155")

        self._redraw_needle()

    def _redraw_needle(self):
        if self._cx is None:
            return
        cx, cy, r = self._cx, self._cy, self._r
        ang = math.radians(180 - self._pct * 180 / 100)
        nx = cx + (r - 20) * math.cos(ang)
        ny = cy - (r - 20) * math.sin(ang)
        if self._needle:
            self.canvas.delete(self._needle)
        self._needle = self.canvas.create_line(cx, cy, nx, ny, width=5,
                                               fill="#111827", capstyle=tk.ROUND)

    def set_value(self, pct: float):
        self._pct = max(0.0, min(100.0, float(pct)))
        if self._pct < 40:
            label, color = f"No Chatter ({self._pct:0.1f}%)", "#16A34A"
        elif self._pct < 70:
            label, color = f"Mild Chatter ({self._pct:0.1f}%)", "#D97706"
        else:
            label, color = f"Heavy Chatter ({self._pct:0.1f}%)", "#DC2626"
        self._status.config(text=label, foreground=color)
        self._redraw_needle()


class BasePage(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, padding=16, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(99, weight=1)

    def headline(self, text):
        lbl = ttk.Label(self, text=text, style="Headline.TLabel")
        lbl.grid(row=0, column=0, sticky="w", pady=(0, 12))
        return lbl

    def placeholder(self, parent, text):
        box = ttk.Label(parent, text=text, style="Placeholder.TLabel",
                        anchor="center", padding=24, relief="ridge")
        box.grid(sticky="nsew")
        return box
