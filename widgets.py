"""
UI widgets
"""
import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from config import CLASS_COLORS, VISIBLE_CLASSES
from data_store import DATA


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


class TrendChart(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.bind("<Configure>", lambda e: self.redraw())

    def redraw(self):
        self.canvas.delete("all")
        w = max(200, self.canvas.winfo_width())
        h = max(140, self.canvas.winfo_height())
        pad = 16
        self.canvas.create_rectangle(0, 0, w, h, fill="white", outline="")
        self.canvas.create_rectangle(pad, pad, w - pad, h - pad, outline="#CBD5E1")

        y = DATA.recent_window(1200)
        if len(y) < 5:
            self.canvas.create_text(w // 2, h // 2, text="(Load CSV to see history)", fill="#6B7280")
            return

        ymin, ymax = min(y), max(y)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0

        def X(i):  return pad + (w - 2 * pad) * (i / (len(y) - 1))
        def Y(v):  return h - pad - (h - 2 * pad) * ((v - ymin) / (ymax - ymin))

        # raw line
        for i in range(1, len(y)):
            self.canvas.create_line(X(i - 1), Y(y[i - 1]), X(i), Y(y[i]), fill="#93C5FD", width=1)

        # smoothed line
        k = max(5, len(y) // 50)
        sm, s = [], 0.0
        for i, v in enumerate(y):
            s += v
            if i >= k: s -= y[i - k]
            sm.append(s / min(i + 1, k))
        for i in range(1, len(sm)):
            self.canvas.create_line(X(i - 1), Y(sm[i - 1]), X(i), Y(sm[i]), fill="#2563EB", width=2)

        # slope badge
        slope = DATA.trend_slope(min(1024, len(y)))
        color = "#DC2626" if slope > 0 else ("#16A34A" if slope < 0 else "#6B7280")
        label = "Uptrend" if slope > 0 else ("Downtrend" if slope < 0 else "Stable")
        self.canvas.create_text(w - pad - 70, pad + 14, text=label, fill=color, font=("Segoe UI", 10, "bold"))
        ax = w - pad - 30; ay = pad + 28
        dy = -16 if slope > 0 else (16 if slope < 0 else 0)
        self.canvas.create_line(ax - 10, ay, ax + 10, ay + dy, arrow=tk.LAST, width=3, fill=color)

        if getattr(DATA, "classes", None):
            # store image references to prevent garbage collection
            if not hasattr(self, '_overlay_images'):
                self._overlay_images = []
            self._overlay_images.clear()

            y0 = pad + 1
            y1 = h - pad - 1
            n_total = len(DATA.od)
            offset = n_total - len(y)

            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                i0 = max(0, min(i0, len(y) - 2))
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                x0 = X(i0)
                x1 = X(i1)

                color = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                rgb = self.canvas.winfo_rgb(color)
                # winfo_rgb returns 16-bit values (0-65535), convert to 8-bit (0-255)
                r = rgb[0] >> 8
                g = rgb[1] >> 8
                b = rgb[2] >> 8
                alpha = 80  # transparency (0=transparent, 255=opaque)

                width = int(x1 - x0)
                height = int(y1 - y0)

                if width > 0 and height > 0:
                    image = Image.new('RGBA', (width, height), (r, g, b, alpha))
                    photo = ImageTk.PhotoImage(image)
                    self._overlay_images.append(photo)
                    self.canvas.create_image(x0, y0, image=photo, anchor='nw')

                # draw label
                self.canvas.create_text(x0 + 4, y0 + 10, text=seg["label"], anchor="w",
                                        fill="#333333", font=("Segoe UI", 8, "bold"))

        # legend
        legend_x = pad + 6
        legend_y = pad + 10
        for idx, (name, col) in enumerate(CLASS_COLORS.items()):
            if name not in VISIBLE_CLASSES: continue
            self.canvas.create_rectangle(legend_x, legend_y + idx * 16 - 6,
                                         legend_x + 12, legend_y + idx * 16 + 6,
                                         fill=col, width=0, stipple="gray25")
            self.canvas.create_text(legend_x + 18, legend_y + idx * 16, text=name, anchor="w",
                                    fill="#111827", font=("Segoe UI", 8))


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
