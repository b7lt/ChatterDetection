import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from widgets import BasePage
from data_store import DATA
from status_bar import status


class ModelPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Select a model and window size")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        left_frame = ttk.Frame(self)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        ttk.Label(left_frame, text="Select Model:", style="Subhead.TLabel").pack(anchor="w", pady=(0, 8))

        self.models = {}
        self.load_models()

        self.cb = ttk.Combobox(left_frame, values=sorted(list(self.models.keys())), height=30, width=40)
        self.cb.set("Pick a model")
        self.cb.pack(pady=(0, 16))
        self.cb.bind('<<ComboboxSelected>>', self.on_model_select)

        ttk.Button(left_frame, text="Update Likelihood Plot", command=self.update_confidence_plot).pack(pady=(8, 0))

        right_frame = ttk.Frame(self)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 0))

        ttk.Label(right_frame,
                  text="Average likelihood of chatter being detected in entire dataset (all windows)\n"
                       "         by each model over different window sizes\n"
                       "Higher likelihood = chatter likely present\n"
                       "Lower likelihood = chatter unlikely\n"
                       "Middle likelihood near decision boundary = unsure prediction",
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
        DATA.model = self.models[self.cb.get()]
        DATA.window_size = int(self.cb.get().split('Window Size ')[1].rstrip(')'))
        if DATA.od:
            DATA.auto_classify(DATA.window_size)
            app_instance = self.winfo_toplevel()
            if hasattr(app_instance, 'pages') and 'Analysis' in app_instance.pages:
                analysis_page = app_instance.pages['Analysis']
                analysis_page.update_confidence_timeline()

            status(f"Model changed and classifications updated")

    def update_confidence_plot(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return

        status("Computing confidence curves... this may take a moment")

        # group models by type and window size
        model_types = {}  # {model_type: {window_size: model}}
        for model_name, model in self.models.items():
            parts = model_name.rsplit(' (Window Size ', 1)
            if len(parts) == 2:
                model_type = parts[0]
                window_size = int(parts[1].rstrip(')'))

                if model_type not in model_types:
                    model_types[model_type] = {}
                model_types[model_type][window_size] = model

        # compute average confidence for each model type at each window size
        results = {}  # {model_type: ([window_sizes], [avg_confidences])}

        for model_type, ws_dict in model_types.items():
            window_sizes = []
            avg_confidences = []

            for ws in sorted(ws_dict.keys()):
                model = ws_dict[ws]

                num_windows = len(DATA.od) // ws
                if num_windows < 1:
                    continue

                features_list = []
                for i in range(num_windows):
                    start_idx = i * ws
                    end_idx = start_idx + ws
                    if end_idx > len(DATA.od):
                        break
                    window = DATA.od[start_idx:end_idx]
                    features_dict = DATA.extract_features(window)
                    features_list.append(features_dict)

                if not features_list:
                    continue

                X = pd.DataFrame(features_list)
                probas = model.predict_proba(X)

                avg_conf = np.mean(probas[:, 1]) * 100

                window_sizes.append(ws)
                avg_confidences.append(avg_conf)

            if window_sizes:
                results[model_type] = (window_sizes, avg_confidences)

        self.ax_conf.clear()

        colors  = ['#2563EB', '#DC2626', '#16A34A', '#F59E0B']
        markers = ['o', 's', '^', 'd']

        for idx, (model_type, (ws, confs)) in enumerate(sorted(results.items())):
            color  = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            self.ax_conf.plot(ws, confs, marker=marker, linewidth=2, markersize=6,
                              label=model_type, color=color)

        self.ax_conf.set_xlabel('Window Size (samples)', fontsize=11)
        self.ax_conf.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax_conf.set_title('Average Chatter Likelihood vs Window Size', fontsize=12, fontweight='bold')
        self.ax_conf.legend(loc='best', fontsize=9)
        self.ax_conf.grid(True, alpha=0.3)
        self.ax_conf.set_ylim([0, 105])

        self.ax_conf.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_conf.text(self.ax_conf.get_xlim()[1], 50, ' Decision boundary',
                          va='center', fontsize=9, color='gray')

        self.fig_conf.tight_layout()
        self.canvas_conf.draw()

        status(f"Confidence plot updated with {len(results)} model types")

    def load_models(self):
        all_model_files = os.listdir("models")
        for model_file in all_model_files:
            if not model_file.endswith('.pkl') or model_file.startswith('scaler_'):
                continue

            model_path = os.path.join("models", model_file)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                name_parts = model_file.split("_")
                match name_parts[1]:
                    case 'logit':
                        model_name = f"Logistic Regression (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'rf':
                        model_name = f"Random Forest (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'svm':
                        model_name = f"SVM (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case 'xgboost':
                        model_name = f"XGBoost (Window Size {name_parts[2][2:-4]})"
                        self.models[model_name] = model
                    case _:
                        self.models[model_file] = model

    def reset_confidence_plot(self):
        self.ax_conf.clear()
        self.ax_conf.text(0.5, 0.5, "Load data and click 'Update Likelihood Plot'\nto see model predictions",
                          ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        status("Confidence plot reset")
