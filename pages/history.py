from tkinter import ttk

from widgets import BasePage, TrendChart


class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Trend")
        ttk.Label(self, foreground="#6B7280").grid(row=1, column=0, sticky="w", pady=(0, 8))

        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=3); grid.columnconfigure(1, weight=2)
        grid.rowconfigure(0, weight=1)

        self.chart = TrendChart(grid)
        self.chart.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        side = ttk.Frame(grid); side.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        side.columnconfigure(0, weight=1)

        self.after(1000, self._tick)

    def _tick(self):
        self.chart.redraw()
        self.after(1000, self._tick)
