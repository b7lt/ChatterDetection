import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from widgets import BasePage
from data_store import DATA
from status_bar import status

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class ModelPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Select Model & Window Size")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        left_frame = ttk.Frame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(left_frame, text="Model Selection (from /models)", style="Subhead.TLabel").pack(anchor="w", pady=(0, 4))

        self.models = {}
        self.load_models()

        self.cb = ttk.Combobox(left_frame, values=sorted(list(self.models.keys())), height=30, width=40)
        self.cb.set("Pick a model")
        self.cb.pack(pady=(0, 8))
        self.cb.bind('<<ComboboxSelected>>', self.on_model_select)

        ws_frame = ttk.Frame(left_frame)
        ws_frame.pack(anchor="w", pady=(0, 12))
        ttk.Label(ws_frame, text="Window Size:").pack(side="left", padx=(0, 6))
        self._ws_var = tk.IntVar(value=2400)
        ttk.Spinbox(ws_frame, textvariable=self._ws_var,
                    from_=240, to=96000, increment=240, width=8).pack(side="left")

        ttk.Button(left_frame, text="Update Likelihood Plot", command=self.update_confidence_plot).pack(pady=(4, 0))

        right_frame = ttk.Frame(self)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(right_frame,
                  text="Average likelihood of chatter being detected in entire dataset (all windows)\n"
                       "Higher likelihood = chatter more likely present on average\n"
                       "Lower likelihood = chatter less likely present on average",
                  style="Subhead.TLabel").pack(anchor="w", pady=(0, 8))

        self.fig_conf = Figure(figsize=(8, 5), dpi=100)
        self.ax_conf = self.fig_conf.add_subplot(111)
        self.canvas_conf = FigureCanvasTkAgg(self.fig_conf, master=right_frame)
        self.canvas_conf.get_tk_widget().pack(fill="both", expand=True)

        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions",
                          ha="center", va="center", fontsize=12)
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()

    def on_model_select(self, event):
        name = self.cb.get()
        DATA.model = self.models[name]
        DATA.window_size = self._ws_var.get()
        if DATA.od:
            DATA.auto_classify(DATA.window_size)
        status(f"Model '{name}' selected, window size={DATA.window_size}")

    def update_confidence_plot(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return
        if not self.models:
            messagebox.showinfo("No Models", "No models loaded from the models/ folder")
            return

        status("Computing confidence curves…")

        ws = self._ws_var.get()
        num_windows = len(DATA.od) // ws
        if num_windows < 1:
            messagebox.showinfo("Window Too Large",
                                f"Window size {ws} is larger than the loaded data.")
            return

        windows = [DATA.od[i * ws:(i + 1) * ws] for i in range(num_windows)]

        results = {}  # {model_name: avg_confidence_pct}
        for model_name, model in self.models.items():
            try:
                old_model = DATA.model
                DATA.model = model
                probas = DATA._cnn_infer(windows)
                DATA.model = old_model
                results[model_name] = float(np.mean(probas[:, 1]) * 100)
            except Exception as exc:
                status(f"Skipped {model_name}: {exc}")

        self.ax_conf.clear()

        names = list(results.keys())
        confs = [results[n] for n in names]
        colors = ['#2563EB', '#DC2626', '#16A34A', '#F59E0B',
                  '#7C3AED', '#DB2777', '#0891B2']

        self.ax_conf.barh(names, confs,
                          color=[colors[i % len(colors)] for i in range(len(names))])
        self.ax_conf.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.6,
                             label='Decision boundary')
        self.ax_conf.set_xlabel('Avg Chatter Likelihood (%)', fontsize=11)
        self.ax_conf.set_title(f'Average Chatter Likelihood  (window={ws:,} samples)',
                               fontsize=12, fontweight='bold')
        self.ax_conf.set_xlim([0, 105])
        self.ax_conf.legend(fontsize=9)
        self.ax_conf.grid(True, axis='x', alpha=0.3)
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()

        status(f"Confidence plot updated — {len(results)} model(s), window={ws:,}")

    def load_models(self):
        if not _TORCH_OK:
            status("PyTorch not available — cannot load CNN models")
            return

        models_dir = "models"
        if not os.path.isdir(models_dir):
            return

        for fname in os.listdir(models_dir):
            if not (fname.endswith('.pt') or fname.endswith('.pth')):
                continue
            path = os.path.join(models_dir, fname)
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
                if callable(getattr(model, 'eval', None)):
                    model.eval()
                    self.models[fname] = model
            except Exception as exc:
                status(f"Could not load {fname}: {exc}")

    def reset_confidence_plot(self):
        self.ax_conf.clear()
        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions",
                          ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        status("Confidence plot reset")
