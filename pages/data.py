import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from widgets import BasePage
from data_store import DATA
from status_bar import status


class DataPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Data")
        controls = ttk.Frame(self); controls.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        for i in range(12): controls.columnconfigure(i, weight=1)

        ttk.Button(controls, text="Load XLSX…", command=self.load_xlsx).grid(row=0, column=0, sticky="w", padx=(0, 8))

        ttk.Label(controls, text="Compare to:").grid(row=0, column=8, sticky="w", padx=(0, 4))
        self.sheet_var = tk.StringVar(value="Select sheet...")
        self.sheet_dropdown = ttk.Combobox(controls, textvariable=self.sheet_var, state="disabled", width=25, height=25)
        self.sheet_dropdown.grid(row=0, column=9, sticky="w", padx=(0, 8))
        self.sheet_dropdown.bind('<<ComboboxSelected>>', self.on_sheet_select)

        ttk.Button(controls, text="Show Correlation", command=self.show_corr).grid(row=0, column=10, sticky="w", padx=(0, 8))

        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

        ttk.Label(controls, text="WebSocket URL:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ws_url = tk.StringVar(value="ws://localhost:6467")
        ttk.Entry(controls, textvariable=self.ws_url, width=40).grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(0, 8))

        ttk.Button(controls, text="Connect Live",  command=self.connect_live).grid(row=1, column=2, sticky="w", pady=(6, 0), padx=(0, 8))
        ttk.Button(controls, text="Disconnect",    command=self.disconnect_live).grid(row=1, column=3, sticky="w", pady=(6, 0))

        self.decimate_var = tk.BooleanVar(value=False)
        def _on_decimate_toggle(*_):
            DATA.decimate_enabled = self.decimate_var.get()
            DATA._decim_current_sec = None
            DATA._decim_vals = []
            DATA._decim_speed_ok = False
            status(f"Decimation to 1 Hz {'enabled' if DATA.decimate_enabled else 'disabled'}")
        self.decimate_var.trace_add('write', _on_decimate_toggle)
        ttk.Checkbutton(controls, text="Decimate live data to 1 Hz (median)", variable=self.decimate_var).grid(
            row=1, column=4, sticky="w", pady=(6, 0), padx=(8, 0))

        self.after(200, self._poll_live_queue)

    def load_xlsx(self):
        path = filedialog.askopenfilename(
            title="Select XLSX file",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if not path: return
        try:
            app_instance = self.winfo_toplevel()
            DATA.load_data(path, app=app_instance)
            self.info.config(text=f"Loaded: {os.path.basename(path)}  •  rows={len(DATA.od)}")

            excluded = ['NDC_System_OD_Value', 'YS_Pullout1_Act_Speed_fpm']
            available = [s for s in DATA.available_sheets if s not in excluded]

            if available:
                self.sheet_dropdown['values'] = available
                self.sheet_dropdown['state'] = 'readonly'
                self.sheet_var.set("Select sheet...")
            else:
                self.sheet_dropdown['values'] = []
                self.sheet_dropdown['state'] = 'disabled'
                self.sheet_var.set("No other sheets")

            app_instance = self.winfo_toplevel()
            if hasattr(app_instance, 'pages') and 'Model' in app_instance.pages:
                app_instance.pages['Model'].reset_confidence_plot()

            status("Data loaded. History & gauge now using real data.")
        except Exception as e:
            messagebox.showerror("Load Data failed", str(e))
            status("Data load failed")

    def on_sheet_select(self, event):
        selected_sheet = self.sheet_var.get()
        if selected_sheet and selected_sheet not in ["Select sheet...", "No other sheets"]:
            try:
                DATA.load_secondary_sheet(selected_sheet)
                self.info.config(text=f"{self.info.cget('text')}  •  comparing to '{selected_sheet}' (paired={len(DATA.paired_df)})")
                status(f"Secondary sheet loaded: {selected_sheet}")
            except Exception as e:
                messagebox.showerror("Load Secondary failed", str(e))
                status("Load secondary failed")
                self.sheet_var.set("Select sheet...")

    def show_corr(self):
        if DATA.paired_df is None or DATA.paired_df.empty:
            messagebox.showinfo("Correlation", "Select a secondary sheet first from the dropdown.")
            return
        from pages.correlation import CorrelationWindow
        CorrelationWindow(self)

    def connect_live(self):
        try:
            DATA.start_live(self.ws_url.get().strip())
            self.info.config(text=f"Live: connected to {self.ws_url.get().strip()}")
            status("Live feed connected")
        except Exception as e:
            messagebox.showerror("Live Connect failed", str(e))
            status("Live connect failed")

    def disconnect_live(self):
        DATA.stop_live()
        self.info.config(text=f"{self.info.cget('text')}  •  live stopped")
        status("Live feed disconnected")

    def _poll_live_queue(self):
        got = DATA._consume_live_queue()
        if got and DATA.model is not None and DATA.window_size and len(DATA.od) >= DATA.window_size:
            now = time.time()
            if not hasattr(self, "_last_cls_ts") or now - self._last_cls_ts >= 1.0:
                DATA.auto_classify(DATA.window_size)
                self._last_cls_ts = now
                app_instance = self.winfo_toplevel()
                if hasattr(app_instance, 'pages') and 'Analysis' in app_instance.pages:
                    analysis_page = app_instance.pages['Analysis']
                    analysis_page.update_confidence_timeline()

        self.after(100, self._poll_live_queue)
