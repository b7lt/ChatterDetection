import asyncio
import concurrent.futures
import json
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from widgets import BasePage
from status_bar import status

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

try:
    from websockets.asyncio.client import connect as ws_connect
except Exception:
    ws_connect = None

# Prefer the fast calamine engine; fall back to openpyxl
try:
    import importlib
    importlib.import_module("python_calamine")
    _XL_ENGINE = "calamine"
except ImportError:
    _XL_ENGINE = "openpyxl"

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SIZE = 2400

LABEL_COLORS = {"chatter": "#EF4444", "no_chatter": "#22C55E"}
LABEL_ALPHA  = 0.25
LISTBOX_BG   = {"chatter": "#FECACA", "no_chatter": "#BBF7D0"}

# Subtle alternating bands per source (cycles if > 6 sources)
_SRC_BANDS = ["#DBEAFE", "#DCFCE7", "#FEF3C7", "#F3E8FF", "#FFE4E6", "#CCFBF1"]

# Known "value" column names in priority order
_VALUE_COLS = ("tag_value", "value", "od", "ovality", "tag_val")


# ── CNN Architecture ─────────────────────────────────────────────────────────
if TORCH_OK:
    class ChatterCNN(nn.Module):
        """
        Input:  (batch, 1, window_size)   — window_size ≥ ~240
        Output: (batch, 3)   softmax → [none, mild, severe]

          Conv1d(1→32, k=16) + BN + ReLU + MaxPool(4)
          Conv1d(32→64, k=8) + BN + ReLU + MaxPool(4)
          Conv1d(64→128, k=4) + BN + ReLU + MaxPool(4)
          Conv1d(128→128, k=4) + BN + ReLU + AdaptiveAvgPool(1)
          Flatten → Dropout(0.4) → Linear(128→64) → ReLU
                  → Dropout(0.2) → Linear(64→3)   → Softmax
        """
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, 16),  nn.BatchNorm1d(32),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 8),  nn.BatchNorm1d(64),  nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 128, 4), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(128, 128, 4), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(64, 3),   nn.Softmax(dim=1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))
else:
    ChatterCNN = None  # type: ignore


# ── Training Page ─────────────────────────────────────────────────────────────
class TrainingPage(BasePage):
    """
    Data model
    ----------
    self._sources  — ordered list of source dicts:
        {
          "name":   str,               display label
          "type":   "xlsx" | "live",
          "series": {col: np.ndarray}, each array has exactly `length` samples
          "length": int,               sample count (= max col length, others NaN-padded)
        }
    Global x-axis = sources concatenated in list order (no physical gaps).
    Offset of source[i] = sum of length for sources[0..i-1].

    self._windows  — list of {"i0": int, "i1": int, "label": str}
        stored in global coordinates; mapped to source-local on training.
        Cross-source-boundary windows are valid labels but skipped during
        training (the source they straddle cannot be determined unambiguously).
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Training")

        self._sources   = []          # list[source dict]
        self._windows   = []          # list[{"i0","i1","label"}]
        self._drag_x    = None        # drag start (xdata)
        self._drag_rect = None        # temp axvspan
        self._model     = None        # trained / imported ChatterCNN
        self._training  = False

        # live feed
        self._live_q    = queue.Queue(maxsize=20_000)
        self._live_stop = threading.Event()
        self._live_thr  = None

        # tkinter vars
        self._label_var = tk.StringVar(value="no_label")
        self._col_var   = tk.StringVar(value="")
        self._ws_size   = tk.IntVar(value=DEFAULT_WINDOW_SIZE)
        self._epochs    = tk.IntVar(value=20)
        self._lr_str    = tk.StringVar(value="0.001")
        self._info_var  = tk.StringVar(value="No data loaded.")
        self._progress  = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.after(200, self._poll_live)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        # ── row 1: controls ──────────────────────────────────────────────────
        ctrl = ttk.Frame(self)
        ctrl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 4))

        # data source
        src_f = ttk.LabelFrame(ctrl, text="Data Source", padding=(6, 3))
        src_f.pack(side="left", padx=(0, 6), fill="y")
        ttk.Button(src_f, text="Import XLSX…", command=self._import_xlsx).grid(
            row=0, column=0, padx=(0, 4))
        ttk.Label(src_f, text="WS URL:").grid(row=0, column=1, padx=(8, 2))
        self._ws_url = tk.StringVar(value="ws://localhost:6467")
        ttk.Entry(src_f, textvariable=self._ws_url, width=24).grid(row=0, column=2)
        ttk.Button(src_f, text="Connect",
                   command=self._connect_ws).grid(row=0, column=3, padx=(4, 0))
        ttk.Button(src_f, text="Disconnect",
                   command=self._disconnect_ws).grid(row=0, column=4, padx=(2, 0))

        # variable selector
        var_f = ttk.LabelFrame(ctrl, text="Plot Variable", padding=(6, 3))
        var_f.pack(side="left", padx=(0, 6), fill="y")
        self._col_combo = ttk.Combobox(var_f, textvariable=self._col_var,
                                        state="readonly", width=24, height=25)
        self._col_combo.pack(pady=2)
        self._col_combo.bind("<<ComboboxSelected>>", lambda _: self._redraw_plot())

        # label selector
        lbl_f = ttk.LabelFrame(ctrl, text="Window Label  (drag on plot)", padding=(6, 3))
        lbl_f.pack(side="left", padx=(0, 6), fill="y")
        for val, txt in [("no_label",   "No Label"),
                         ("no_chatter", "No Chatter"),
                         ("chatter",    "Chatter")]:
            ttk.Radiobutton(lbl_f, text=txt, variable=self._label_var,
                            value=val).pack(side="left", padx=6)

        # model / training
        mdl_f = ttk.LabelFrame(ctrl, text="Model & Training", padding=(6, 3))
        mdl_f.pack(side="left", fill="y")
        c = 0
        ttk.Button(mdl_f, text="Import Model…",
                   command=self._import_model).grid(row=0, column=c, padx=(0, 6)); c += 1
        ttk.Label(mdl_f, text="Win sz:").grid(row=0, column=c, padx=(0, 2)); c += 1
        ttk.Spinbox(mdl_f, textvariable=self._ws_size,
                    from_=240, to=19200, increment=240, width=7).grid(
            row=0, column=c, padx=(0, 6)); c += 1
        ttk.Label(mdl_f, text="Epochs:").grid(row=0, column=c, padx=(0, 2)); c += 1
        ttk.Spinbox(mdl_f, textvariable=self._epochs,
                    from_=1, to=1000, width=5).grid(
            row=0, column=c, padx=(0, 6)); c += 1
        ttk.Label(mdl_f, text="LR:").grid(row=0, column=c, padx=(0, 2)); c += 1
        ttk.Entry(mdl_f, textvariable=self._lr_str, width=8).grid(
            row=0, column=c, padx=(0, 6)); c += 1
        self._train_btn = ttk.Button(mdl_f, text="Train",
                                     command=self._start_training, state="disabled")
        self._train_btn.grid(row=0, column=c, padx=(0, 4)); c += 1
        self._save_btn = ttk.Button(mdl_f, text="Save Model…",
                                    command=self._save_model, state="disabled")
        self._save_btn.grid(row=0, column=c)

        # ── row 2: info ──────────────────────────────────────────────────────
        ttk.Label(self, textvariable=self._info_var, foreground="#6B7280").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # ── row 3: progress (hidden) ─────────────────────────────────────────
        self._prog_lbl = ttk.Label(self, text="")
        self._prog_lbl.grid(row=3, column=0, columnspan=2, sticky="w")
        self._prog_bar = ttk.Progressbar(self, variable=self._progress, maximum=100)
        self._prog_bar.grid(row=3, column=0, columnspan=2, sticky="ew")
        self._prog_bar.grid_remove()
        self._prog_lbl.grid_remove()

        # ── row 4: plot + right panel ─────────────────────────────────────────
        self.rowconfigure(4, weight=1)

        plot_frm = ttk.Frame(self)
        plot_frm.grid(row=4, column=0, sticky="nsew")
        plot_frm.rowconfigure(0, weight=1)
        plot_frm.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(12, 5), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frm)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        tb_frm = ttk.Frame(plot_frm)
        tb_frm.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frm)
        self._toolbar.update()

        self._canvas.mpl_connect("button_press_event",   self._on_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._init_plot()

        # ── right panel ───────────────────────────────────────────────────────
        right = ttk.Frame(self)
        right.grid(row=4, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Sources sub-panel
        src_panel = ttk.LabelFrame(right, text="Sources", padding=(6, 4))
        src_panel.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        src_panel.columnconfigure(0, weight=1)

        src_list_frm = ttk.Frame(src_panel)
        src_list_frm.grid(row=0, column=0, sticky="ew")
        src_list_frm.columnconfigure(0, weight=1)

        self._src_lb = tk.Listbox(src_list_frm, width=28, height=5,
                                   font=("Segoe UI", 9), selectmode="extended")
        src_sb = ttk.Scrollbar(src_list_frm, orient="vertical",
                                command=self._src_lb.yview)
        self._src_lb.config(yscrollcommand=src_sb.set)
        self._src_lb.grid(row=0, column=0, sticky="ew")
        src_sb.grid(row=0, column=1, sticky="ns")

        ttk.Button(src_panel, text="Remove Selected",
                   command=self._remove_selected_sources).grid(
            row=1, column=0, sticky="ew", pady=(4, 0))

        # Windows sub-panel
        win_panel = ttk.LabelFrame(right, text="Labeled Windows", padding=(6, 4))
        win_panel.grid(row=1, column=0, sticky="nsew")
        win_panel.rowconfigure(0, weight=1)
        win_panel.columnconfigure(0, weight=1)

        wlf = ttk.Frame(win_panel)
        wlf.grid(row=0, column=0, sticky="nsew")
        wlf.rowconfigure(0, weight=1)
        wlf.columnconfigure(0, weight=1)

        self._win_lb = tk.Listbox(wlf, width=26, selectmode="extended",
                                   font=("Courier New", 9))
        win_sb = ttk.Scrollbar(wlf, orient="vertical", command=self._win_lb.yview)
        self._win_lb.config(yscrollcommand=win_sb.set)
        self._win_lb.grid(row=0, column=0, sticky="nsew")
        win_sb.grid(row=0, column=1, sticky="ns")

        ttk.Button(win_panel, text="Delete Selected",
                   command=self._delete_selected_windows).grid(
            row=1, column=0, sticky="ew", pady=(4, 1))
        ttk.Button(win_panel, text="Clear All",
                   command=self._clear_all_windows).grid(
            row=2, column=0, sticky="ew", pady=1)

        self._count_lbl = ttk.Label(win_panel, text="", foreground="#374151",
                                     font=("Segoe UI", 9))
        self._count_lbl.grid(row=3, column=0, sticky="w", pady=(4, 0))

    # ── Source helpers ────────────────────────────────────────────────────────
    def _offset_of(self, idx: int) -> int:
        """Global start index of source at list position idx."""
        return sum(s["length"] for s in self._sources[:idx])

    def _total_length(self) -> int:
        return sum(s["length"] for s in self._sources)

    def _source_for_window(self, i0: int, i1: int):
        """Return (source_dict, src_offset) if [i0,i1) lies fully inside one source."""
        off = 0
        for s in self._sources:
            if i0 >= off and i1 <= off + s["length"]:
                return s, off
            off += s["length"]
        return None, -1

    def _get_combined(self, col: str) -> np.ndarray:
        """Concatenate `col` across all sources; NaN where a source lacks it."""
        parts = []
        for s in self._sources:
            if col in s["series"]:
                arr = np.asarray(s["series"][col], dtype=np.float32)
            else:
                arr = np.full(s["length"], np.nan, dtype=np.float32)
            # defensive: pad if array is shorter than declared length
            if len(arr) < s["length"]:
                arr = np.concatenate([arr,
                    np.full(s["length"] - len(arr), np.nan, dtype=np.float32)])
            parts.append(arr[:s["length"]])
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def _rebuild_col_combo(self):
        all_cols = sorted({c for s in self._sources for c in s["series"]})
        self._col_combo["values"] = all_cols
        if self._col_var.get() not in all_cols and all_cols:
            self._col_var.set(all_cols[0])

    def _refresh_src_list(self):
        self._src_lb.delete(0, tk.END)
        for i, s in enumerate(self._sources):
            tag   = "[WS]" if s["type"] == "live" else "[XL]"
            entry = f"{tag} {s['name']}  ({s['length']:,} smp)"
            self._src_lb.insert(tk.END, entry)
            bg = _SRC_BANDS[i % len(_SRC_BANDS)]
            self._src_lb.itemconfig(i, background=bg)

    def _remove_selected_sources(self):
        idxs = sorted(self._src_lb.curselection(), reverse=True)
        if not idxs:
            return
        # Process highest index first so earlier offsets stay valid
        for idx in idxs:
            off    = self._offset_of(idx)
            length = self._sources[idx]["length"]

            # Discard or shift windows
            kept = []
            for w in self._windows:
                if w["i1"] <= off or w["i0"] >= off + length:
                    # outside: shift if it comes after the removed source
                    if w["i0"] >= off + length:
                        kept.append({"i0": w["i0"] - length,
                                     "i1": w["i1"] - length,
                                     "label": w["label"]})
                    else:
                        kept.append(w)
                # overlapping → discard
            self._windows = kept

            if self._sources[idx]["type"] == "live":
                self._live_stop.set()
                self._live_thr = None

            del self._sources[idx]

        self._rebuild_col_combo()
        self._refresh_src_list()
        self._refresh_win_list()
        self._redraw_plot()
        self._check_train_ready()
        self._update_info()

    def _update_info(self):
        n = len(self._sources)
        t = self._total_length()
        lw = sum(1 for w in self._windows if w["label"] in LABEL_COLORS)
        self._info_var.set(
            f"{n} source(s)  •  {t:,} total samples  •  {lw} labeled window(s)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    def _init_plot(self):
        self._ax.clear()
        self._ax.text(0.5, 0.5, "Import XLSX files or connect WebSocket to begin",
                      ha="center", va="center", fontsize=13, color="#9CA3AF",
                      transform=self._ax.transAxes)
        self._ax.axis("off")
        self._canvas.draw_idle()

    def _redraw_plot(self):
        col = self._col_var.get()
        if not self._sources or not col:
            return

        combined = self._get_combined(col)
        if len(combined) == 0:
            return

        self._ax.clear()
        self._ax.set_visible(True)

        # ── time-series line ─────────────────────────────────────────────────
        xs = np.arange(len(combined))
        self._ax.plot(xs, combined, linewidth=0.7, color="#3B82F6",
                      alpha=0.85, rasterized=True)

        # ── per-source band, separator, and label ────────────────────────────
        off = 0
        for i, s in enumerate(self._sources):
            x0, x1 = off, off + s["length"]
            band_c  = _SRC_BANDS[i % len(_SRC_BANDS)]

            if x1 > x0:
                self._ax.axvspan(x0, x1, alpha=0.08, color=band_c, linewidth=0, zorder=0)

            # vertical separator (not before first source)
            if i > 0:
                self._ax.axvline(x0, color="#64748B", linewidth=1.6,
                                 linestyle="--", alpha=0.85, zorder=3)

            # source label at top (axes-y = 1.0 means top edge)
            has = col in s["series"]
            marker = "✓" if has else "✗ (no data)"
            label  = f"{s['name']}\n{marker}"
            mid    = x0 + s["length"] / 2
            self._ax.text(
                mid, 1.0, label,
                transform=self._ax.get_xaxis_transform(),
                ha="center", va="bottom",
                fontsize=7, color="#1E293B",
                bbox=dict(boxstyle="round,pad=0.25", fc=band_c,
                          ec="#CBD5E1", alpha=0.90, linewidth=0.8),
                clip_on=True,
            )
            off += s["length"]

        # ── labeled window overlays ───────────────────────────────────────────
        for w in self._windows:
            c = LABEL_COLORS.get(w["label"])
            if c:
                self._ax.axvspan(w["i0"], w["i1"], alpha=LABEL_ALPHA,
                                 color=c, linewidth=0, zorder=2)

        self._ax.set_xlabel("Global sample index", fontsize=10)
        self._ax.set_ylabel(col, fontsize=10)
        n_src = len(self._sources)
        self._ax.set_title(
            f"{col}  –  {n_src} source(s), {self._total_length():,} samples  |  "
            f"win sz = {self._ws_size.get():,}",
            fontsize=10)
        self._ax.grid(True, alpha=0.2, linewidth=0.5)
        self._fig.tight_layout(pad=1.5)
        self._canvas.draw_idle()

    # ── Mouse interaction ─────────────────────────────────────────────────────
    def _toolbar_active(self) -> bool:
        try:
            return bool(self._toolbar.mode)
        except Exception:
            return False

    def _on_press(self, event):
        if event.inaxes is not self._ax or self._toolbar_active():
            return
        if event.button == 1:
            self._drag_x = event.xdata

    def _on_drag(self, event):
        if self._drag_x is None:
            return
        if event.inaxes is not self._ax:
            return

        curr = event.xdata if event.xdata is not None else self._drag_x
        x0, x1 = min(self._drag_x, curr), max(self._drag_x, curr)

        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except Exception:
                pass
            self._drag_rect = None

        color = LABEL_COLORS.get(self._label_var.get(), "#94A3B8")
        self._drag_rect = self._ax.axvspan(x0, x1, alpha=0.12, color=color, linewidth=0)
        self._canvas.draw_idle()

    def _on_release(self, event):
        if self._drag_x is None:
            return

        if self._drag_rect is not None:
            try:
                self._drag_rect.remove()
            except Exception:
                pass
            self._drag_rect = None

        x_end = (event.xdata
                 if (event.inaxes is self._ax and event.xdata is not None)
                 else self._drag_x)
        x0 = min(self._drag_x, x_end)
        x1 = max(self._drag_x, x_end)
        self._drag_x = None

        if not self._sources:
            return

        n   = self._total_length()
        ws  = self._ws_size.get()
        lbl = self._label_var.get()

        w_start = max(0, (int(x0) // ws) * ws)
        w_end   = min(n, (int(x1) // ws + 1) * ws)
        if w_end <= w_start:
            self._redraw_plot()
            return

        count = 0
        for i0 in range(w_start, w_end, ws):
            self._set_window(i0, min(i0 + ws, n), lbl)
            count += 1

        self._refresh_win_list()
        self._redraw_plot()
        self._check_train_ready()
        action = "Cleared" if lbl == "no_label" else f"Labeled ({lbl.replace('_', ' ')})"
        status(f"{action}: {count} window(s)")

    def _set_window(self, i0: int, i1: int, label: str):
        self._windows = [w for w in self._windows
                         if not (w["i0"] == i0 and w["i1"] == i1)]
        if label != "no_label":
            self._windows.append({"i0": i0, "i1": i1, "label": label})
        self._windows.sort(key=lambda w: w["i0"])

    def _refresh_win_list(self):
        self._win_lb.delete(0, tk.END)
        sym = {"chatter": "C", "no_chatter": "N"}
        for i, w in enumerate(self._windows):
            s = sym.get(w["label"], "?")
            self._win_lb.insert(tk.END, f"[{s}] {w['i0']:>9} .. {w['i1']:<9}")
            self._win_lb.itemconfig(i, background=LISTBOX_BG.get(w["label"], "#FFFFFF"))
        nc = sum(1 for w in self._windows if w["label"] == "no_chatter")
        ch = sum(1 for w in self._windows if w["label"] == "chatter")
        self._count_lbl.config(text=f"No Chatter: {nc}   Chatter: {ch}   Total: {nc+ch}")

    def _delete_selected_windows(self):
        for i in sorted(self._win_lb.curselection(), reverse=True):
            del self._windows[i]
        self._refresh_win_list()
        self._redraw_plot()
        self._check_train_ready()

    def _clear_all_windows(self):
        if self._windows and not messagebox.askyesno("Clear All",
                                                      "Remove all labeled windows?"):
            return
        self._windows.clear()
        self._refresh_win_list()
        self._redraw_plot()
        self._check_train_ready()

    # ── XLSX import (fast, multi-file, background) ────────────────────────────
    def _import_xlsx(self):
        paths = filedialog.askopenfilenames(
            title="Import XLSX file(s) for Training",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not paths:
            return
        status(f"Loading {len(paths)} file(s) using {_XL_ENGINE}…")
        self.update_idletasks()
        threading.Thread(target=self._xlsx_load_thread,
                         args=(list(paths),), daemon=True).start()

    def _xlsx_load_thread(self, paths: list):
        # Read files in parallel (each opens its own file handle – safe)
        workers = min(4, len(paths))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(self._parse_one_xlsx, paths))
        self.after(0, self._on_xlsx_done, results)

    @staticmethod
    def _parse_one_xlsx(path: str) -> dict:
        """
        Parse a single XLSX and return a source dict or an error dict.
        Runs in a worker thread – must not touch tkinter.
        """
        name   = os.path.basename(path)
        series = {}

        try:
            # Open the file once; iterate sheets without re-reading the file
            with pd.ExcelFile(path, engine=_XL_ENGINE) as xl:
                for sheet in xl.sheet_names:
                    try:
                        vals = TrainingPage._read_sheet_values(xl, sheet)
                        if vals is not None and len(vals) > 0:
                            series[sheet] = vals
                    except Exception:
                        pass
        except Exception as exc:
            return {"error": str(exc), "name": name}

        if not series:
            return {"error": "No numeric data found in any sheet.", "name": name}

        length = max(len(v) for v in series.values())

        # Pad shorter columns with NaN so all arrays equal `length`
        for col, arr in series.items():
            if len(arr) < length:
                series[col] = np.concatenate(
                    [arr, np.full(length - len(arr), np.nan, dtype=np.float32)])

        return {"name": name, "type": "xlsx", "series": series, "length": length}

    @staticmethod
    def _read_sheet_values(xl: pd.ExcelFile, sheet: str):
        """
        Try the fast path (known column names + dtype hint) first.
        Fall back to reading the full sheet and finding any numeric column.
        Returns np.ndarray[float32] or None.
        """
        # Fast path: only read the two columns we know about
        try:
            df = xl.parse(sheet,
                          usecols=["t_stamp", "Tag_value"],
                          dtype={"Tag_value": np.float32},
                          parse_dates=False)
            return df["Tag_value"].dropna().to_numpy(dtype=np.float32)
        except Exception:
            pass  # column names differ – fall through

        # Fallback: read full sheet (parse_dates=False for speed)
        try:
            df = xl.parse(sheet, parse_dates=False)
        except Exception:
            return None

        if df.empty:
            return None

        # Priority list of value column names
        col_lower = {c.lower(): c for c in df.columns}
        for name in _VALUE_COLS:
            if name in col_lower:
                return (pd.to_numeric(df[col_lower[name]], errors="coerce")
                          .dropna()
                          .to_numpy(dtype=np.float32))

        # Last resort: first numeric-dtype column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return df[c].dropna().to_numpy(dtype=np.float32)

        return None

    def _on_xlsx_done(self, results: list):
        errors, added = [], 0
        for r in results:
            if "error" in r:
                errors.append(f"{r['name']}: {r['error']}")
            else:
                self._sources.append(r)
                added += 1

        if errors:
            messagebox.showerror("Import Errors", "\n".join(errors))

        if added:
            self._rebuild_col_combo()
            self._refresh_src_list()
            self._redraw_plot()
            self._check_train_ready()
            self._update_info()
            status(f"Loaded {added} file(s)  (engine: {_XL_ENGINE})")
        else:
            status("No files loaded.")

    # ── WebSocket live feed ───────────────────────────────────────────────────
    def _connect_ws(self):
        if ws_connect is None:
            messagebox.showerror("WebSocket",
                                 "websockets package not installed.\n"
                                 "pip install websockets")
            return
        if self._live_thr and self._live_thr.is_alive():
            messagebox.showinfo("Already Connected", "Live feed is already running.")
            return

        # Reuse or create the live source (always the last source for clean layout)
        live_src = next((s for s in self._sources if s["type"] == "live"), None)
        if live_src is None:
            live_src = {"name": "Live (WS)", "type": "live",
                        "series": {}, "length": 0}
            self._sources.append(live_src)
        else:
            # Reset buffer so replay doesn't split the source
            live_src["series"] = {}
            live_src["length"] = 0

        self._rebuild_col_combo()
        self._refresh_src_list()

        url = self._ws_url.get().strip()
        self._live_stop.clear()
        self._live_thr = threading.Thread(
            target=lambda: asyncio.run(self._live_main(url)), daemon=True)
        self._live_thr.start()
        self._info_var.set(f"Live: connected to {url}  •  collecting…")
        status("Training live feed connected")

    def _disconnect_ws(self):
        self._live_stop.set()
        self._info_var.set(self._info_var.get() + "  •  disconnected")
        status("Training live feed disconnected")

    async def _live_main(self, url: str):
        while not self._live_stop.is_set():
            try:
                async with ws_connect(url) as ws:
                    while not self._live_stop.is_set():
                        msg  = await ws.recv()
                        data = json.loads(msg)
                        # Server sends batched messages: {"samples": [{...}, ...]}
                        # Fall back to single-sample format for backwards compatibility.
                        items = data.get("samples") or [data]
                        for item in items:
                            # Speed filter: drop samples where line is stopped
                            speed = item.get("YS_Pullout1_Act_Speed_fpm")
                            if speed is not None and speed <= 1:
                                continue
                            # Collect NDC_System_OD_Value (and any other numeric fields)
                            numeric = {
                                k: float(v)
                                for k, v in item.items()
                                if k != "t_stamp" and isinstance(v, (int, float))
                                and not math.isnan(float(v))
                            }
                            if numeric:
                                try:
                                    self._live_q.put_nowait(numeric)
                                except queue.Full:
                                    pass
            except Exception:
                await asyncio.sleep(0.5)

    def _poll_live(self):
        batch = []
        while True:
            try:
                batch.append(self._live_q.get_nowait())
            except queue.Empty:
                break

        if batch:
            live_src = next((s for s in self._sources if s["type"] == "live"), None)
            if live_src is not None:
                prev_keys = set(live_src["series"])
                # Fan each message's values into per-key series lists
                for msg_dict in batch:
                    for key, val in msg_dict.items():
                        if key not in live_src["series"]:
                            live_src["series"][key] = []
                        live_src["series"][key].append(val)
                # Length = longest series (shorter ones will NaN-pad in _get_combined)
                if live_src["series"]:
                    live_src["length"] = max(len(v) for v in live_src["series"].values())
                n = live_src["length"]
                # Rebuild combo if new keys appeared
                if set(live_src["series"]) != prev_keys:
                    self._rebuild_col_combo()
                # Batch redraws (~every 500 new samples)
                if n % 500 < len(batch):
                    self._refresh_src_list()
                    self._redraw_plot()
                    self._update_info()

        self.after(200, self._poll_live)

    # ── Model import ──────────────────────────────────────────────────────────
    def _import_model(self):
        if not TORCH_OK:
            messagebox.showerror("PyTorch",
                                 "PyTorch not installed.\npip install torch")
            return
        path = filedialog.askopenfilename(
            title="Import CNN Model",
            filetypes=[("PyTorch model", "*.pt *.pth"),
                       ("Pickle", "*.pkl"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        try:
            if path.endswith(".pkl"):
                import pickle
                with open(path, "rb") as f:
                    self._model = pickle.load(f)
            else:
                self._model = torch.load(path, map_location="cpu", weights_only=False)
            self._save_btn.config(state="normal")
            status(f"Model imported: {os.path.basename(path)}")
            messagebox.showinfo("Import", f"Model loaded:\n{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Import Failed", str(e))

    # ── Training ──────────────────────────────────────────────────────────────
    def _check_train_ready(self):
        has_data = self._total_length() > 0
        has_lbl  = any(w["label"] in LABEL_COLORS for w in self._windows)
        state    = "normal" if (TORCH_OK and has_data and has_lbl) else "disabled"
        self._train_btn.config(state=state)

    def _start_training(self):
        if self._training:
            return
        if not TORCH_OK:
            messagebox.showerror("PyTorch",
                                 "PyTorch not installed.\npip install torch")
            return

        col = self._col_var.get()
        ws  = self._ws_size.get()

        LABEL_MAP = {"no_chatter": 0, "chatter": 1}
        X_list, y_list = [], []
        skipped_cross, skipped_no_col = 0, 0

        for w in self._windows:
            if w["label"] not in LABEL_MAP:
                continue
            src, src_off = self._source_for_window(w["i0"], w["i1"])
            if src is None:
                skipped_cross += 1
                continue
            if col not in src["series"]:
                skipped_no_col += 1
                continue

            local_i0 = w["i0"] - src_off
            local_i1 = w["i1"] - src_off
            seg = np.asarray(src["series"][col][local_i0:local_i1], dtype=np.float32)

            # pad / trim to window size
            if len(seg) < ws:
                seg = np.pad(seg, (0, ws - len(seg)), constant_values=np.nan)
            else:
                seg = seg[:ws]

            # replace NaN with channel mean before normalizing
            nan_mask = np.isnan(seg)
            if nan_mask.all():
                skipped_no_col += 1
                continue
            seg[nan_mask] = np.nanmean(seg)

            mu, sigma = seg.mean(), seg.std()
            seg = (seg - mu) / (sigma + 1e-8)
            X_list.append(seg)
            y_list.append(LABEL_MAP[w["label"]])

        if not X_list:
            parts = ["No valid labeled windows for training."]
            if skipped_cross:
                parts.append(f"  • {skipped_cross} window(s) cross source boundaries")
            if skipped_no_col:
                parts.append(f"  • {skipped_no_col} window(s) missing variable '{col}'")
            messagebox.showwarning("No Data", "\n".join(parts))
            return

        try:
            lr = float(self._lr_str.get())
        except ValueError:
            messagebox.showerror("Bad LR", "Learning rate must be a float (e.g. 0.001)")
            return

        skip_msg = ""
        if skipped_cross or skipped_no_col:
            skip_msg = (f"  ({skipped_cross} cross-boundary, "
                        f"{skipped_no_col} missing-variable skipped)")

        X = torch.tensor(np.stack(X_list), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y_list, dtype=torch.long)

        self._training = True
        self._train_btn.config(state="disabled")
        self._prog_bar.grid()
        self._prog_lbl.grid()
        self._progress.set(0)
        status(f"Training on {len(X_list)} windows{skip_msg}…")

        threading.Thread(target=self._train_thread,
                         args=(X, y, self._epochs.get(), lr),
                         daemon=True).start()

    def _train_thread(self, X, y, epochs, lr):
        try:
            device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model    = ChatterCNN().to(device)

            if self._model is not None:
                try:
                    model.load_state_dict(self._model.state_dict())
                except Exception:
                    pass  # architecture mismatch – train from scratch

            ds      = TensorDataset(X.to(device), y.to(device))
            loader  = DataLoader(ds, batch_size=min(32, len(y)),
                                 shuffle=True, drop_last=False)

            crit    = nn.CrossEntropyLoss()
            opt     = optim.Adam(model.parameters(), lr=lr)
            sched   = optim.lr_scheduler.StepLR(
                opt, step_size=max(1, epochs // 5), gamma=0.5)

            model.train()
            for epoch in range(epochs):
                total_loss, correct, total = 0.0, 0, 0
                for xb, yb in loader:
                    opt.zero_grad()
                    out  = model(xb)
                    loss = crit(out, yb)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    correct    += (out.argmax(1) == yb).sum().item()
                    total      += len(yb)
                sched.step()

                pct = (epoch + 1) / epochs * 100
                acc = correct / total * 100 if total else 0.0
                self.after(0, self._update_progress, pct,
                           f"Epoch {epoch+1}/{epochs}  "
                           f"loss={total_loss/len(loader):.4f}  acc={acc:.1f}%")

            self._model = model.cpu()
            self.after(0, self._training_done)

        except Exception as e:
            self.after(0, self._training_error, str(e))

    def _update_progress(self, pct, msg):
        self._progress.set(pct)
        self._prog_lbl.config(text=msg)

    def _training_done(self):
        self._training = False
        self._progress.set(100)
        self._prog_lbl.config(text="Training complete!")
        self._train_btn.config(state="normal")
        self._save_btn.config(state="normal")
        status("CNN training complete")
        messagebox.showinfo("Training", "Training complete!\nUse 'Save Model…' to export.")

    def _training_error(self, msg):
        self._training = False
        self._train_btn.config(state="normal")
        self._prog_bar.grid_remove()
        self._prog_lbl.grid_remove()
        status("Training failed")
        messagebox.showerror("Training Error", msg)

    # ── Save model ────────────────────────────────────────────────────────────
    def _save_model(self):
        if self._model is None:
            messagebox.showwarning("No Model", "No trained model to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save CNN Model",
            defaultextension=".pt",
            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            torch.save(self._model, path)
            status(f"Model saved: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Model saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Failed", str(e))
