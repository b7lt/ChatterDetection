import math
from datetime import datetime

from tkinter import ttk

from widgets import BasePage, Gauge
from pages.live import LiveTimeSeries
from data_store import DATA


class LivePage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Live Data & Predictions")

        grid = ttk.Frame(self); grid.grid(row=1, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(1, weight=1)
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1, minsize=220)
        grid.rowconfigure(0, weight=1)

        left = ttk.Frame(grid); left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(left, text="Live OD", style="Subhead.TLabel").pack(anchor="w")
        self.live = LiveTimeSeries(left)
        self.live.pack(fill="both", expand=True, pady=(6, 0))

        right = ttk.Frame(grid); right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(right, text="Chatter Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right)
        self.gauge.pack(fill="x", pady=(6, 0))

        self.after(1000, self._tick)

    def _tick(self):
        pct = 50

        if DATA.od and DATA.classes:
            lbl, class_risk = DATA.current_class()
            if class_risk is not None:
                pct = class_risk * 100.0
            else:
                pct = 0
        else:
            pct = 50 + 15 * math.sin(datetime.now().timestamp() / 2.0)

        self.gauge.set_value(pct)
        self.after(1000, self._tick)
