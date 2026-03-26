import tkinter as tk
from tkinter import ttk
from datetime import datetime

import status_bar
from config import APP_TITLE, APP_VERSION
from pages.data import DataPage
from pages.training import TrainingPage
from pages.model import ModelPage
from pages.live_page import LivePage
from pages.history import HistoryPage

class App(tk.Tk):
    _status_var = None

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x850"); self.minsize(1120, 720)

        self._init_style(); self._init_menu()

        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0); root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        sidebar = self._build_sidebar(root); sidebar.grid(row=0, column=0, sticky="nsw")
        self.container = ttk.Frame(root, padding=(12, 16, 16, 16)); self.container.grid(row=0, column=1, sticky="nsew")
        self.container.columnconfigure(0, weight=1); self.container.rowconfigure(0, weight=1)

        self.pages = {
            "Data":     DataPage(self.container),
            "Training": TrainingPage(self.container),
            "Model":    ModelPage(self.container),
            "Live":      LivePage(self.container),
            "History":  HistoryPage(self.container),
        }
        for p in self.pages.values(): p.grid(row=0, column=0, sticky="nsew")
        self.show("Live")

        self._build_statusbar()
        self.bind("<Control-Key-1>", lambda e: self.show("Data"))
        self.bind("<Control-Key-2>", lambda e: self.show("Training"))
        self.bind("<Control-Key-3>", lambda e: self.show("Model"))
        self.bind("<Control-Key-4>", lambda e: self.show("Live"))
        self.bind("<Control-Key-5>", lambda e: self.show("History"))

    def _init_style(self):
        self.style = ttk.Style(self)
        try: self.style.theme_use("clam")
        except tk.TclError: pass
        self.style.configure("Sidebar.TFrame",      background="#111827")
        self.style.configure("Sidebar.TButton",     foreground="white", background="#1F2937")
        self.style.map("Sidebar.TButton",            background=[("active", "#374151")])
        self.style.configure("Headline.TLabel",     font=("Segoe UI", 18, "bold"))
        self.style.configure("Subhead.TLabel",      font=("Segoe UI", 12, "bold"))
        self.style.configure("Placeholder.TLabel",  foreground="#6B7280", background="white")
        self.style.configure("KPI.TLabel",          font=("Segoe UI", 10, "bold"))

    def _init_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_sidebar(self, parent):
        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        ttk.Label(bar, text="Chatter Detection", foreground="white", background="#111827",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0, 16))
        for name in ["Data", "Training", "Model", "Live", "History"]:
            ttk.Button(bar, text=name, style="Sidebar.TButton",
                       command=lambda n=name: self.show(n)).pack(fill="x", pady=6)
        ttk.Label(bar, text="", background="#111827").pack(expand=True, fill="both")
        ttk.Label(bar, text=f"{APP_VERSION}", foreground="#9CA3AF", background="#111827").pack(anchor="w")
        return bar

    def _build_statusbar(self):
        App._status_var = tk.StringVar(value="Ready")
        status_bar._var = App._status_var   # wire up the global status bridge
        bar = ttk.Frame(self); bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=App._status_var, padding=8).pack(side="left")
        ttk.Label(bar, text=datetime.now().strftime("%Y-%m-%d"), padding=8).pack(side="right")

    def show(self, page_name: str):
        self.pages[page_name].tkraise(); self.status(f"Showing {page_name}")

    @classmethod
    def status(cls, msg: str):
        status_bar.status(msg)

    @classmethod
    def busy(cls, msg: str):
        cls.status(msg)


if __name__ == "__main__":
    App().mainloop()
