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
    def __init__(self, parent, width=360, height=200, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.width, self.height = width, height
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self._needle = None
        self._pct_label = ttk.Label(self, text="— %", font=("Segoe UI", 12, "bold"))
        self._pct_label.grid(row=1, column=0, pady=(8, 0))
        self._status = ttk.Label(self, text="Status: —", font=("Segoe UI", 11))
        self._status.grid(row=2, column=0)
        self._draw_static()
        self.set_value(0)

    def _draw_static(self):
        w, h = self.width, self.height
        cx, cy, r = w // 2, h - 10, min(w, h * 2) // 2 - 10
        self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=180, extent=-60, fill="#16A34A", outline="")
        self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=120, extent=-60, fill="#D97706", outline="")
        self.canvas.create_arc(cx - r, cy - r, cx + r, cy + r, start=60,  extent=-60, fill="#DC2626", outline="")
        for i in range(0, 11):
            ang = math.radians(180 + i * 18)
            x0 = cx + (r - 18) * math.cos(ang); y0 = cy + (r - 18) * math.sin(ang)
            x1 = cx + (r - 2)  * math.cos(ang); y1 = cy + (r - 2)  * math.sin(ang)
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill="#334155")
        self.canvas.create_text(cx - r + 40, cy - 20, text="OK", font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(cx + r - 40, cy - 20, text="NG", font=("Segoe UI", 11, "bold"))
        self._cx, self._cy, self._r = cx, cy, r

    def set_value(self, pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        ang = math.radians(180 + 180 * pct / 100.0)
        cx, cy, r = self._cx, self._cy, self._r
        x = cx + (r - 26) * math.cos(ang)
        y = cy + (r - 26) * math.sin(ang)
        if self._needle: self.canvas.delete(self._needle)
        self._needle = self.canvas.create_line(cx, cy, x, y, width=5, fill="#111827", capstyle=tk.ROUND)
        self._pct_label.config(text=f"{pct:0.1f}% confidence")
        status_text = "NG" if pct >= 50 else "OK"
        self._status.config(text=f"Status: {status_text}",
                            foreground=("#DC2626" if status_text == "NG" else "#16A34A"))


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
