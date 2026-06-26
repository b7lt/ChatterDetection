import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from widgets import BasePage
from data_store import DATA, ModelInputError
from status_bar import status

try:
    import torch
    _TORCH_OK = True
    _TORCH_IMPORT_ERROR = ""
except ImportError as exc:
    _TORCH_OK = False
    _TORCH_IMPORT_ERROR = str(exc)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "models"


class ModelPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Select Model & Window Size")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        left_frame = ttk.Frame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(left_frame, text="Model Selection", style="Subhead.TLabel").pack(anchor="w", pady=(0, 4))

        self.models = {}
        self._model_paths = {}

        load_frame = ttk.Frame(left_frame)
        load_frame.pack(anchor="w", fill="x", pady=(0, 8))
        ttk.Button(load_frame, text="Import Model Files...",
                   command=self.import_model_files).pack(side="left", padx=(0, 6))
        ttk.Button(load_frame, text="Import Folder...",
                   command=self.import_model_folder).pack(side="left", padx=(0, 6))
        ttk.Button(load_frame, text="Clear",
                   command=self.clear_models).pack(side="left")

        self.cb = ttk.Combobox(left_frame, values=sorted(list(self.models.keys())),
                               state="readonly", height=30, width=40)
        self.cb.set("Import one or more models")
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
        if name not in self.models:
            return
        DATA.model = self.models[name]
        DATA.window_size = self._ws_var.get()
        if DATA.od:
            try:
                DATA.auto_classify(DATA.window_size, raise_errors=True)
            except ModelInputError as exc:
                messagebox.showerror("Model Input Mismatch", str(exc))
                status(f"Model '{name}' selected but input dimensions do not match")
                return
        status(f"Model '{name}' selected, window size={DATA.window_size}")

    def update_confidence_plot(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return
        if not self.models:
            messagebox.showinfo("No Models", "Import one or more model files first.")
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
        old_model = DATA.model
        for model_name, model in self.models.items():
            try:
                DATA.model = model
                ctx = None
                if DATA._is_hybrid_model():
                    ctx = [
                        DATA._build_context_vector(i * ws, (i + 1) * ws)
                        for i in range(num_windows)
                    ]
                probas = DATA._cnn_infer(windows, ctx)
                results[model_name] = float(np.mean(probas[:, 1]) * 100)
            except ModelInputError as exc:
                status(f"Skipped {model_name}: {exc}")
            except Exception as exc:
                status(f"Skipped {model_name}: {exc}")
            finally:
                DATA.model = old_model

        self.ax_conf.clear()

        if not results:
            self.ax_conf.text(0.5, 0.5, "No imported models produced predictions",
                              ha="center", va="center", fontsize=12, color="#6B7280")
            self.ax_conf.axis("off")
            self.fig_conf.tight_layout()
            self.canvas_conf.draw()
            status("Confidence plot skipped — no valid model predictions")
            return

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

    def import_model_files(self):
        paths = filedialog.askopenfilenames(
            title="Import Model File(s)",
            initialdir=self._initial_model_dir(),
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")]
        )
        if paths:
            self._load_model_paths(paths)

    def import_model_folder(self):
        folder = filedialog.askdirectory(
            title="Import Models From Folder",
            initialdir=self._initial_model_dir(),
        )
        if not folder:
            return
        paths = []
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".pt", ".pth")):
                    paths.append(os.path.join(root, fname))
        if not paths:
            messagebox.showinfo("No Models", "No .pt or .pth files were found in that folder.")
            return
        self._load_model_paths(sorted(paths))

    def clear_models(self):
        self.models.clear()
        self._model_paths.clear()
        self._refresh_model_selector()
        DATA.model = None
        DATA._classified_with_model = None
        DATA.classes = []
        status("Imported models cleared")

    def _load_model_paths(self, paths):
        if not _TORCH_OK:
            self._show_torch_missing()
            return

        loaded = 0
        errors = []
        for path in paths:
            if not path.lower().endswith((".pt", ".pth")):
                continue
            abs_path = os.path.abspath(path)
            if abs_path in self._model_paths.values():
                continue
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
                if callable(getattr(model, 'eval', None)):
                    model.eval()
                    label = self._display_name_for_path(abs_path)
                    self.models[label] = model
                    self._model_paths[label] = abs_path
                    loaded += 1
                else:
                    errors.append(f"{os.path.basename(path)}: not a torch model")
            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        self._refresh_model_selector()

        if errors:
            messagebox.showwarning("Some Models Could Not Be Loaded", "\n".join(errors))
        if loaded:
            if DATA.model is None and self.cb.get() in self.models:
                DATA.model = self.models[self.cb.get()]
                DATA.window_size = self._ws_var.get()
            status(f"Imported {loaded} model(s)")
        elif not errors:
            status("No new models imported")

    def _display_name_for_path(self, path: str) -> str:
        name = os.path.basename(path)
        if name not in self.models:
            return name

        parent_name = os.path.basename(os.path.dirname(path))
        candidate = f"{parent_name}/{name}" if parent_name else name
        if candidate not in self.models:
            return candidate

        stem, ext = os.path.splitext(candidate)
        idx = 2
        while f"{stem} ({idx}){ext}" in self.models:
            idx += 1
        return f"{stem} ({idx}){ext}"

    def _refresh_model_selector(self):
        names = sorted(self.models)
        self.cb["values"] = names
        if names:
            current = self.cb.get()
            self.cb.set(current if current in self.models else names[0])
        else:
            self.cb.set("Import one or more models")

    def _initial_model_dir(self):
        return str(_DEFAULT_MODELS_DIR if _DEFAULT_MODELS_DIR.is_dir() else _PROJECT_ROOT)

    def _show_torch_missing(self):
        msg = (
            "PyTorch is not installed for the Python interpreter running this dashboard.\n\n"
            f"Interpreter:\n{sys.executable}\n\n"
            "Install PyTorch in that environment, then restart the dashboard:\n"
            "python -m pip install torch\n\n"
            "If this works in VSCode, make sure your terminal is using the same interpreter."
        )
        if _TORCH_IMPORT_ERROR:
            msg += f"\n\nImport error: {_TORCH_IMPORT_ERROR}"
        messagebox.showerror("PyTorch Required", msg)
        status("PyTorch not available — cannot load CNN models")

    def reset_confidence_plot(self):
        self.ax_conf.clear()
        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions",
                          ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        status("Confidence plot reset")
