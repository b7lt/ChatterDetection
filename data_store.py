import asyncio
import importlib
import json
import math
import os
import queue
import threading
import time

import numpy as np
import pandas as pd

try:
    from websockets.asyncio.client import connect as ws_connect
except Exception:
    ws_connect = None

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from config import (
    FOOTAGE_TAG,
    HISTORY_SAMPLE_LIMIT,
    LIVE_QUEUE_DRAIN_LIMIT,
    LIVE_QUEUE_MAX,
    LIVE_SAMPLE_LIMIT,
    LIVE_SAMPLE_TRIM_TO,
    OD_TAG,
    PRESSURE_TAG,
    RUNNING_FOOTAGE_MIN,
    RUNNING_PRESSURE_MIN,
    RUNNING_SPEED_MIN,
    SECONDARY_COL_GUESSES,
    SEGMENT_FOOTAGE_RESET_DROP,
    SEGMENT_IDLE_AFTER_SECONDS,
    SPEED_TAG_CANDIDATES,
    pick,
)
from status_bar import status
from models import CONTEXT_TAGS


try:
    importlib.import_module("python_calamine")
    _XL_ENGINE = "calamine"
except ImportError:
    _XL_ENGINE = "openpyxl"


class ModelInputError(ValueError):
    """Raised when a selected model cannot consume the current dashboard inputs."""


class DataStore:
    def __init__(self):
        self.path = None
        self.ts = []
        self.ts_dt = []
        self.od = []
        self.od_hist = []       # 1 Hz downsampled OD, capped separately
        self.ts_hist = []       # pd.Timestamp per od_hist sample
        self.ctx_hist: dict = {tag: [] for tag in CONTEXT_TAGS}
                                # 1 Hz downsampled context, parallel to od_hist/ts_hist
        self.classes = []

        self.paired_df = None

        # Context signal storage — parallel to self.od (same length, same trim)
        self.context_data: dict = {tag: [] for tag in CONTEXT_TAGS}

        # Latest values of every field arriving over WebSocket, for KPI display
        self.live_snapshot: dict = {}
        self.segment_state = "idle"
        self.current_segment_id = 0
        self.segment_started_at = None
        self.segment_last_seen_at = None
        self.segment_start_footage = None
        self.segment_current_footage = None
        self.segment_current_pressure = None
        self._last_segment_footage = None

        self.model = None
        self.window_size = None
        self._classified_with_model = None
        self.last_inference_error = ""
        self._blocked_inference_key = None

        self.available_sheets = []

        self.live_url = "ws://localhost:6467"
        self.live_thread = None
        self.live_stop = threading.Event()
        self.live_queue = queue.Queue(maxsize=LIVE_QUEUE_MAX)
        self.live_dropped = 0
        self.live_rejected = 0

        self._decim_current_sec = None
        self._decim_vals = []
        self._decim_ctx_vals: dict = {}
        self._decim_speed_ok = False

        # 1 Hz historical buffer accumulators
        self._hist_current_sec = None
        self._hist_vals = []
        self._hist_ctx_vals: dict = {}

        self.target_hz = 1
        self.decimate_enabled = False

        self._trim_offset = 0
        self._MAX_LIVE = LIVE_SAMPLE_LIMIT
        self._TRIM_TO = LIVE_SAMPLE_TRIM_TO
        self._MAX_HISTORY = HISTORY_SAMPLE_LIMIT

    # ── File I/O ──────────────────────────────────────────────────────────────

    @staticmethod
    def _excel_file(path: str):
        return pd.ExcelFile(path, engine=_XL_ENGINE)

    @staticmethod
    def _read_excel(path: str, sheet_name, **kwargs):
        return pd.read_excel(path, sheet_name=sheet_name, engine=_XL_ENGINE, **kwargs)

    def _read_any_table(self, path: str, sheet=None):
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            if sheet is None:
                return self._read_excel(path, sheet_name=None, parse_dates=[0])
            else:
                with self._excel_file(path) as xl:
                    available = set(xl.sheet_names)
                sheets_to_read = [sheet]
                for tag in self._segment_signal_sheets(available) + list(CONTEXT_TAGS):
                    if tag in available:
                        sheets_to_read.append(tag)
                seen = set()
                unique = [s for s in sheets_to_read if not (s in seen or seen.add(s))]
                return self._read_excel(path, sheet_name=unique, parse_dates=[0])

    def _smart_to_numeric(self, series):
        s = series.copy()
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        s = s.astype(str).str.strip()
        s = s.str.replace(r"[^\d\.,\-eE+]", "", regex=True)
        has_comma = s.str.contains(",", regex=False, na=False).sum()
        has_dot   = s.str.contains(".", regex=False, na=False).sum()
        if has_comma > has_dot:
            s = s.str.replace(".", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
        else:
            s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _segment_signal_sheets(available: set | None = None):
        names = list(SPEED_TAG_CANDIDATES) + [FOOTAGE_TAG, PRESSURE_TAG]
        if available is not None:
            names = [name for name in names if name in available]
        seen = set()
        return [name for name in names if not (name in seen or seen.add(name))]

    @staticmethod
    def _first_available(names, mapping):
        return next((name for name in names if name in mapping), None)

    def _running_indices(self, df_dict):
        masks = []

        speed_tag = self._first_available(SPEED_TAG_CANDIDATES, df_dict)
        if speed_tag:
            speed = self._smart_to_numeric(df_dict[speed_tag]["Tag_value"])
            masks.append(speed > RUNNING_SPEED_MIN)

        if FOOTAGE_TAG in df_dict:
            footage = self._smart_to_numeric(df_dict[FOOTAGE_TAG]["Tag_value"])
            footage_delta = footage.diff().fillna(0)
            masks.append((footage > RUNNING_FOOTAGE_MIN) | (footage_delta > 0))

        if PRESSURE_TAG in df_dict:
            pressure = self._smart_to_numeric(df_dict[PRESSURE_TAG]["Tag_value"])
            masks.append(pressure > RUNNING_PRESSURE_MIN)

        if not masks:
            return None

        valid = masks[0].fillna(False)
        for mask in masks[1:]:
            valid = valid | mask.fillna(False)
        return valid[valid].index

    def filter_by_running_state(self, df_dict):
        valid_indices = self._running_indices(df_dict)
        if valid_indices is None:
            return {
                sheet_name: df.reset_index(drop=True)
                for sheet_name, df in df_dict.items()
                if len(df) > 0
            }

        filtered = {}
        for sheet_name, df in df_dict.items():
            if len(df) == 0:
                continue
            filtered[sheet_name] = df.reindex(valid_indices).reset_index(drop=True)
        return filtered

    def load_data(self, path: str, app=None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path

        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            with self._excel_file(path) as xl:
                self.available_sheets = xl.sheet_names
        else:
            self.available_sheets = []

        df_dict = self._read_any_table(path, OD_TAG)
        if df_dict is None or len(df_dict) == 0:
            raise ValueError("Empty file.")

        filtered = self.filter_by_running_state(df_dict)

        self.od    = filtered[OD_TAG]["Tag_value"].tolist()
        self.ts    = filtered[OD_TAG]["t_stamp"].astype(str).tolist()
        self.ts_dt = pd.to_datetime(
            filtered[OD_TAG]["t_stamp"], errors="coerce").tolist()
        self.classes      = []
        self._trim_offset = 0

        # ── Context data from sheet tabs ──────────────────────────────────────
        self.context_data = {tag: [] for tag in CONTEXT_TAGS}
        n_od = len(self.od)
        for tag in CONTEXT_TAGS:
            if tag in filtered:
                df_ctx = filtered[tag]
                vals = (df_ctx["Tag_value"].tolist()
                        if "Tag_value" in df_ctx.columns else [])
                if len(vals) < n_od:
                    vals = vals + [0.0] * (n_od - len(vals))
                self.context_data[tag] = vals[:n_od]
            else:
                self.context_data[tag] = [0.0] * n_od

        # ── 1 Hz OD historical buffer ─────────────────────────────────────────
        self.od_hist = []
        self.ts_hist = []
        self._hist_current_sec = None
        self._hist_vals = []
        self._hist_ctx_vals = {}
        try:
            df_h = pd.DataFrame({"t": self.ts_dt, "od": self.od})
            df_h["t"] = pd.to_datetime(df_h["t"])
            df_h = df_h.set_index("t").resample("1s").median().dropna().reset_index()
            self.od_hist = df_h["od"].tolist()
            self.ts_hist = df_h["t"].tolist()
        except Exception:
            pass

        # ── 1 Hz context historical buffer (resampled from context_data) ──────
        self.ctx_hist = {tag: [] for tag in CONTEXT_TAGS}
        try:
            if self.ts_dt and len(self.ts_dt) == len(self.od):
                df_ctx = pd.DataFrame({"t": pd.to_datetime(self.ts_dt)})
                for tag in CONTEXT_TAGS:
                    vals = self.context_data.get(tag, [])
                    if len(vals) == len(self.od):
                        df_ctx[tag] = vals
                df_ctx = df_ctx.set_index("t")
                df_r   = df_ctx.resample("1s").median().reset_index()
                n = len(self.ts_hist)
                for tag in CONTEXT_TAGS:
                    if tag in df_r.columns:
                        self.ctx_hist[tag] = df_r[tag].tolist()[:n]
                    else:
                        self.ctx_hist[tag] = [0.0] * n
        except Exception:
            pass
        self._trim_history()

        try:
            v = self.od
            status(f"Loaded: rows={len(v)}  min={min(v):.4g}  "
                   f"max={max(v):.4g}  mean={sum(v)/len(v):.4g}")
        except Exception:
            pass

        if self.model is not None and app is not None:
            self.auto_classify(window_size=self.window_size)

    # ── Secondary / correlation ────────────────────────────────────────────────

    def _align_series(self, df_main, tcol_main, ycol_main, df_sec, tcol_sec, ycol_sec):
        m = pd.DataFrame({
            "t":  pd.to_datetime(df_main[tcol_main], errors="coerce"),
            "od": self._smart_to_numeric(df_main[ycol_main]),
        }).dropna()
        s = pd.DataFrame({
            "t":   pd.to_datetime(df_sec[tcol_sec], errors="coerce"),
            "sec": self._smart_to_numeric(df_sec[ycol_sec]),
        }).dropna()
        for df in [m, s]:
            try:   df["t"] = df["t"].dt.tz_convert(None)
            except: df["t"] = df["t"].dt.tz_localize(None)
        m = m.sort_values("t")
        s = s.sort_values("t")
        tol = pd.Timedelta(seconds=1)
        if len(m) >= 3:
            dtm = m["t"].diff().dropna().median() or pd.Timedelta(seconds=1)
            tol = max(tol, dtm)
        if len(s) >= 3:
            dts = s["t"].diff().dropna().median() or pd.Timedelta(seconds=1)
            tol = max(tol, dts)
        paired = pd.merge_asof(m, s, on="t", direction="nearest", tolerance=tol)
        return paired.dropna().reset_index(drop=True)[["t", "od", "sec"]]

    def load_secondary_sheet(self, sheet_name: str):
        if not self.path:
            raise ValueError("Load the main data file first.")
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        ext = os.path.splitext(self.path.lower())[1]
        if ext not in [".xlsx", ".xls"]:
            raise ValueError("Secondary sheet loading only works with Excel files.")

        with self._excel_file(self.path) as xl:
            available = set(xl.sheet_names)
        sheets = [sheet_name, OD_TAG] + self._segment_signal_sheets(available)
        seen = set()
        sheets = [s for s in sheets if s in available and not (s in seen or seen.add(s))]
        df_dict = self._read_excel(self.path, sheet_name=sheets, parse_dates=[0])
        df_sec = df_dict[sheet_name]
        if df_sec is None or df_sec.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")

        filtered = self.filter_by_running_state(df_dict)
        df_sec_f = filtered[sheet_name]
        cols_s   = list(df_sec_f.columns)
        tcol_s   = pick(cols_s, SECONDARY_COL_GUESSES["time"]) or "t_stamp"
        ycol_s   = pick(cols_s, SECONDARY_COL_GUESSES["val"])  or "Tag_value"
        if tcol_s not in cols_s:
            raise ValueError(f"No time column in '{sheet_name}'. Columns: {cols_s}")
        if ycol_s not in cols_s:
            raise ValueError(f"No value column in '{sheet_name}'. Columns: {cols_s}")
        df_main_f = filtered[OD_TAG]
        paired = self._align_series(
            df_main_f, "t_stamp", "Tag_value", df_sec_f, tcol_s, ycol_s)
        if paired.empty:
            raise ValueError(
                f"No overlapping timestamps between OD and '{sheet_name}'.")
        self.paired_df = paired
        status(f"Secondary loaded: {sheet_name} * paired rows={len(paired)}")

    def corr_stats(self, max_lag_samples: int = 300):
        if self.paired_df is None or self.paired_df.empty:
            return {"n": 0, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}
        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 3:
            return {"n": n, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}
        r0 = float(np.corrcoef(x, y)[0, 1])
        best_r, best_k = r0, 0
        for k in range(1, min(max_lag_samples, n - 2) + 1):
            r_pos = float(np.corrcoef(x[k:], y[:-k])[0, 1])
            if r_pos > best_r: best_r, best_k = r_pos, +k
            r_neg = float(np.corrcoef(x[:-k], y[k:])[0, 1])
            if r_neg > best_r: best_r, best_k = r_neg, -k
        return {"n": n, "pearson_r": r0, "best_lag": best_k, "r_at_best_lag": best_r}

    def _paired_ok(self):
        return self.paired_df is not None and not self.paired_df.empty

    def lag_corr_curve(self, max_lag_samples=300):
        if not self._paired_ok(): return np.array([]), np.array([])
        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 5: return np.array([]), np.array([])
        K    = int(min(max_lag_samples, n - 3))
        lags = np.arange(-K, K + 1, dtype=int)
        r    = np.zeros_like(lags, dtype=float)
        for i, k in enumerate(lags):
            if k < 0:   r[i] = np.corrcoef(x[:k],  y[-k:])[0, 1]
            elif k > 0: r[i] = np.corrcoef(x[k:],  y[:-k])[0, 1]
            else:       r[i] = np.corrcoef(x,       y     )[0, 1]
        return lags, r

    def rolling_corr(self, win_samples=200, step=10):
        if not self._paired_ok(): return np.array([]), np.array([])
        df = self.paired_df
        x  = df["od"].to_numpy(dtype=float)
        y  = df["sec"].to_numpy(dtype=float)
        t  = pd.to_datetime(df["t"], errors="coerce").to_numpy()
        n  = len(df)
        if n < max(10, win_samples): return np.array([]), np.array([])
        mids, rr = [], []
        for i0 in range(0, n - win_samples + 1, step):
            i1 = i0 + win_samples
            sx, sy = x[i0:i1], y[i0:i1]
            r = (float(np.corrcoef(sx, sy)[0, 1])
                 if np.std(sx) > 1e-12 and np.std(sy) > 1e-12 else np.nan)
            mids.append(t[i0 + win_samples // 2])
            rr.append(r)
        return np.array(mids), np.array(rr)

    # ── Classification ────────────────────────────────────────────────────────

    def current_class(self):
        if not self.classes:
            return None, None
        return self.classes[-1]["label"], self.classes[-1]["risk"]

    def _is_hybrid_model(self) -> bool:
        return getattr(self.model, "is_hybrid", False) or hasattr(self.model, "mlp")

    def _build_context_vector(self, local_start: int, local_end: int) -> list:
        vec = []
        for tag in CONTEXT_TAGS:
            data = self.context_data.get(tag, [])
            seg  = data[local_start:local_end]
            if seg:
                vals = [v for v in seg if not (isinstance(v, float) and np.isnan(v))]
                vec.append(float(np.mean(vals)) if vals else 0.0)
            else:
                vec.append(0.0)
        return vec

    def _validate_context_shape(self, context_list):
        if context_list is None:
            raise ModelInputError(
                "The selected model expects context inputs, but no context "
                "vectors were available for inference."
            )

        ctx = np.asarray(context_list, dtype=np.float32)
        if ctx.ndim != 2:
            raise ModelInputError(
                f"Context input must be 2-D, got shape {ctx.shape}."
            )

        model_mean = getattr(self.model, "ctx_mean", None)
        model_std = getattr(self.model, "ctx_std", None)
        if model_mean is None or model_std is None:
            raise ModelInputError(
                "The selected hybrid model is missing ctx_mean/ctx_std. "
                "Retrain or resave it with the current training pipeline."
            )

        ctx_mean = np.asarray(model_mean, dtype=np.float32).reshape(-1)
        ctx_std = np.asarray(model_std, dtype=np.float32).reshape(-1)
        if len(ctx_mean) != len(ctx_std):
            raise ModelInputError(
                f"The selected model has inconsistent context normalizer sizes "
                f"(ctx_mean={len(ctx_mean)}, ctx_std={len(ctx_std)})."
            )

        actual = ctx.shape[1]
        expected = len(ctx_mean)
        if actual != expected:
            tags = ", ".join(CONTEXT_TAGS)
            raise ModelInputError(
                "The selected model was trained with a different number of "
                f"context variables. Model expects {expected}; the dashboard "
                f"is providing {actual}. Current context tags: {tags}. Use a "
                "model trained with this context tag set, or update CONTEXT_TAGS "
                "to match the model."
            )

        return ctx, ctx_mean, ctx_std

    def _cnn_infer(self, windows_list, context_list=None):
        import torch
        self.model.eval()
        segs = []
        for w in windows_list:
            seg = np.asarray(w, dtype=np.float32)
            mu, sigma = seg.mean(), seg.std()
            seg = (seg - mu) / (sigma + 1e-8)
            segs.append(seg)
        X = torch.tensor(np.stack(segs), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            if self._is_hybrid_model():
                ctx, ctx_mean, ctx_std = self._validate_context_shape(context_list)
                ctx      = (ctx - ctx_mean) / (ctx_std + 1e-8)
                X_ctx    = torch.tensor(ctx, dtype=torch.float32)
                return self.model(X, X_ctx).numpy()
            else:
                return self.model(X).numpy()

    def get_label_from_risk_prob(self, risk):
        if   risk < 0.40: return "No Chatter"
        elif risk < 0.70: return "Mild Chatter"
        else:             return "Heavy Chatter"

    def auto_classify(self, window_size=60, raise_errors=False):
        if self.model is None:
            status("No model selected."); return
        if not window_size or window_size <= 0:
            status("Invalid window size."); return
        if len(self.od) < window_size:
            return

        inference_key = (id(self.model), int(window_size))
        if not raise_errors and self._blocked_inference_key == inference_key:
            return

        total_abs   = len(self.od) + self._trim_offset
        num_windows = total_abs // window_size

        if self.classes and (
            self.classes[0]["i1"] - self.classes[0]["i0"] != window_size
            or self._classified_with_model is not self.model
        ):
            self.classes = []

        if self.classes:
            next_window = max(c["i1"] for c in self.classes) // window_size
        else:
            next_window = (self._trim_offset + window_size - 1) // window_size

        if next_window >= num_windows:
            return

        windows, ctx_list, meta = [], [], []
        use_hybrid = self._is_hybrid_model()

        for i in range(next_window, num_windows):
            abs_s = i * window_size
            abs_e = abs_s + window_size
            loc_s = abs_s - self._trim_offset
            loc_e = abs_e - self._trim_offset
            if loc_s < 0 or loc_e > len(self.od):
                continue
            windows.append(self.od[loc_s:loc_e])
            if use_hybrid:
                ctx_list.append(self._build_context_vector(loc_s, loc_e))
            meta.append((abs_s, abs_e, loc_s, loc_e))

        if not windows:
            return

        try:
            probas = self._cnn_infer(
                windows,
                ctx_list if use_hybrid else None
            )
            self.last_inference_error = ""
            self._blocked_inference_key = None
        except (ModelInputError, RuntimeError, ValueError) as exc:
            self.last_inference_error = str(exc)
            self._blocked_inference_key = inference_key
            status(f"Model input mismatch: {exc}")
            if raise_errors:
                if isinstance(exc, ModelInputError):
                    raise
                raise ModelInputError(str(exc)) from exc
            return

        for i, (abs_s, abs_e, loc_s, loc_e) in enumerate(meta):
            risk = float(probas[i][1])
            self.classes.append({
                "start": self.ts[loc_s],
                "end":   self.ts[loc_e - 1],
                "label": self.get_label_from_risk_prob(risk),
                "i0":    abs_s,
                "i1":    abs_e,
                "risk":  risk,
            })

        self._classified_with_model = self.model
        status(f"Auto-classes: {len(self.classes)}")

    # ── Explainability ────────────────────────────────────────────────────────

    def explain_last_window(self) -> dict | None:
        """
        Gradient x input attribution for the MLP context branch of the most
        recently classified window.

        Returns None if:
          - model is not HybridChatterNet
          - no windows classified yet
          - PyTorch unavailable

        Return dict:
            tags          list[str]    CONTEXT_TAGS
            raw_values    list[float]  un-normalised per-window feature means
            attributions  list[float]  signed: positive pushes toward chatter
            grads         list[float]  raw gradient (sign = direction of risk)
            chatter_prob  float        model chatter probability for this window
        """
        if not _TORCH_OK:
            return None
        if not self._is_hybrid_model() or not self.classes or not self.od:
            return None

        try:
            import torch

            last  = self.classes[-1]
            loc_s = last["i0"] - self._trim_offset
            loc_e = last["i1"] - self._trim_offset
            if loc_s < 0 or loc_e > len(self.od):
                return None

            # OD tensor — no gradient tracking needed for the CNN branch
            seg = np.asarray(self.od[loc_s:loc_e], dtype=np.float32)
            mu, sigma = seg.mean(), seg.std()
            seg = (seg - mu) / (sigma + 1e-8)
            X   = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Context tensor — we want d(chatter_prob)/d(X_ctx)
            ctx_vec  = self._build_context_vector(loc_s, loc_e)
            ctx      = np.array(ctx_vec, dtype=np.float32)
            ctx_mean = np.asarray(self.model.ctx_mean, dtype=np.float32)
            ctx_std  = np.asarray(self.model.ctx_std,  dtype=np.float32)
            ctx_norm = (ctx - ctx_mean) / (ctx_std + 1e-8)

            X_ctx = torch.tensor(ctx_norm, dtype=torch.float32).unsqueeze(0)
            X_ctx.requires_grad_(True)

            self.model.eval()
            # Must NOT use torch.no_grad() here — we need the autograd graph for X_ctx
            out = self.model(X, X_ctx)
            out[0, 1].backward()   # differentiate chatter probability

            grads        = X_ctx.grad[0].detach().numpy().copy()
            attributions = (grads * ctx_norm).tolist()  # gradient x input

            self.model.zero_grad()  # housekeeping

            return {
                "tags":         list(CONTEXT_TAGS),
                "raw_values":   ctx_vec,
                "attributions": attributions,
                "grads":        grads.tolist(),
                "chatter_prob": float(out[0, 1].detach()),
            }
        except Exception:
            return None

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _linreg_slope(y):
        n = len(y)
        if n < 2: return 0.0
        sx  = n * (n - 1) / 2.0
        sxx = n * (n - 1) * (2 * n - 1) / 6.0
        sy  = sum(y)
        sxy = sum(i * yi for i, yi in enumerate(y))
        d   = n * sxx - sx * sx
        if abs(d) < 1e-12: return 0.0
        return (n * sxy - sx * sy) / d

    def recent_window(self, n=1024):
        if not self.od: return []
        return self.od[-min(n, len(self.od)):]

    def trend_slope(self, n=1024):
        y = self.recent_window(n)
        return self._linreg_slope(y) if y else 0.0

    # ── Sample append ─────────────────────────────────────────────────────────

    def _append_sample(self, ts_dt, od, context=None):
        self.ts_dt.append(ts_dt)
        self.ts.append(str(ts_dt))
        self.od.append(float(od))
        for tag in CONTEXT_TAGS:
            lst = self.context_data.setdefault(tag, [])
            lst.append(float(context[tag]) if context and tag in context else 0.0)

        if len(self.od) > self._MAX_LIVE:
            excess = len(self.od) - self._TRIM_TO
            del self.od[:excess]
            del self.ts[:excess]
            del self.ts_dt[:excess]
            for lst in self.context_data.values():
                del lst[:excess]
            self._trim_offset += excess
            self._trim_class_windows()

    def _append_history_sample(self, ts_sec: int, od_values: list, ctx_values: dict):
        self.od_hist.append(float(np.median(od_values)))
        self.ts_hist.append(pd.to_datetime(ts_sec, unit="s"))
        for tag in CONTEXT_TAGS:
            vals = ctx_values.get(tag, [])
            self.ctx_hist.setdefault(tag, []).append(
                float(np.median(vals)) if vals else np.nan)
        self._trim_history()

    def _trim_class_windows(self):
        if self.classes:
            self.classes = [
                span for span in self.classes
                if span.get("i1", 0) > self._trim_offset
            ]

    def _trim_history(self):
        if self._MAX_HISTORY <= 0 or len(self.od_hist) <= self._MAX_HISTORY:
            return
        excess = len(self.od_hist) - self._MAX_HISTORY
        del self.od_hist[:excess]
        del self.ts_hist[:excess]
        for vals in self.ctx_hist.values():
            del vals[:excess]

    # ── Segment tracking ─────────────────────────────────────────────────────

    def _start_segment(self, ts: float, footage):
        self.current_segment_id += 1
        self.segment_state = "running"
        self.segment_started_at = ts
        self.segment_last_seen_at = ts
        self.segment_start_footage = footage

    def _stop_segment(self):
        self.segment_state = "idle"
        self.segment_last_seen_at = None
        self.segment_start_footage = None

    def _update_segment_state(self, ts: float, signals: dict) -> bool:
        speed = self._first_available_value(SPEED_TAG_CANDIDATES, signals)
        footage = signals.get(FOOTAGE_TAG)
        pressure = signals.get(PRESSURE_TAG)

        footage_reset = (
            footage is not None
            and self._last_segment_footage is not None
            and footage + SEGMENT_FOOTAGE_RESET_DROP < self._last_segment_footage
        )

        if footage_reset:
            self._stop_segment()

        evidence = False
        if speed is not None:
            evidence = speed > RUNNING_SPEED_MIN
        else:
            if footage is not None:
                if self._last_segment_footage is None:
                    evidence = footage > RUNNING_FOOTAGE_MIN
                else:
                    evidence = (footage - self._last_segment_footage) > 0
            if pressure is not None:
                last_pressure = self.segment_current_pressure
                pressure_rising = (
                    last_pressure is not None
                    and (pressure - last_pressure) > 0
                )
                evidence = evidence or pressure_rising
                if self.segment_state == "idle":
                    evidence = evidence or pressure > RUNNING_PRESSURE_MIN

        if evidence:
            if self.segment_state != "running":
                self._start_segment(ts, footage)
            else:
                self.segment_last_seen_at = ts
        elif (
            self.segment_state == "running"
            and self.segment_last_seen_at is not None
            and ts - self.segment_last_seen_at >= SEGMENT_IDLE_AFTER_SECONDS
        ):
            self._stop_segment()

        if footage is not None:
            self.segment_current_footage = footage
            self._last_segment_footage = footage
        if pressure is not None:
            self.segment_current_pressure = pressure

        return self.segment_state == "running"

    @staticmethod
    def _first_available_value(names, mapping):
        for name in names:
            value = mapping.get(name)
            if value is not None:
                return value
        return None

    def _segment_snapshot_values(self, signals: dict):
        snapshot = {
            "segment_state": self.segment_state,
            "current_segment_id": self.current_segment_id,
        }
        snapshot.update(signals)
        return snapshot

    # ── Live queue consumer ────────────────────────────────────────────────────

    def _consume_live_queue(self):
        drained = 0
        while drained < LIVE_QUEUE_DRAIN_LIMIT:
            try:
                item = self.live_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            ts, od, signals, ctx = item
            running = self._update_segment_state(ts, signals)

            # Update live snapshot for KPI display
            self.live_snapshot.update(ctx)
            self.live_snapshot[OD_TAG] = od
            for tag, value in self._segment_snapshot_values(signals).items():
                self.live_snapshot[tag] = value

            # Always maintain 1 Hz historical buffer (OD + context)
            hist_sec = int(ts)
            if self._hist_current_sec is None:
                self._hist_current_sec = hist_sec

            if hist_sec != self._hist_current_sec:
                if self._hist_vals:
                    self._append_history_sample(
                        self._hist_current_sec,
                        self._hist_vals,
                        self._hist_ctx_vals,
                    )
                self._hist_current_sec = hist_sec
                self._hist_vals        = []
                self._hist_ctx_vals    = {}

            if running:
                self._hist_vals.append(od)
                if ctx:
                    for tag in CONTEXT_TAGS:
                        if tag in ctx:
                            self._hist_ctx_vals.setdefault(tag, []).append(ctx[tag])

            # Non-decimated path
            if not getattr(self, "decimate_enabled", False):
                if not running:
                    continue
                self._append_sample(pd.to_datetime(ts, unit="s"), od, ctx)
                self._decim_current_sec = None
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False
                continue

            # Decimated path (~1 Hz median)
            sec = int(ts)
            if self._decim_current_sec is None:
                self._decim_current_sec = sec
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False

            if sec != self._decim_current_sec:
                if self._decim_speed_ok and self._decim_vals:
                    ctx_med = {
                        tag: float(np.median(v)) if v else 0.0
                        for tag, v in self._decim_ctx_vals.items()
                    }
                    self._append_sample(
                        pd.to_datetime(self._decim_current_sec, unit="s"),
                        float(np.median(self._decim_vals)),
                        ctx_med,
                    )
                self._decim_current_sec = sec
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False

            if running:
                self._decim_speed_ok = True
                self._decim_vals.append(od)
                for tag, val in ctx.items():
                    self._decim_ctx_vals.setdefault(tag, []).append(val)

        return drained

    # ── WebSocket live feed ────────────────────────────────────────────────────

    @staticmethod
    def _as_float(value, default=None):
        if value is None:
            return default
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return out if math.isfinite(out) else default

    @staticmethod
    def _timestamp_seconds(value, fallback: float):
        if value in (None, ""):
            return fallback
        numeric = DataStore._as_float(value)
        if numeric is not None:
            # Treat large millisecond epochs as milliseconds, otherwise seconds.
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        try:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(parsed):
                return fallback
            return parsed.timestamp()
        except Exception:
            return fallback

    def _live_segment_signals(self, item: dict, ctx: dict):
        signals = {}
        for tag in self._segment_signal_sheets():
            value = self._as_float(item.get(tag), default=None)
            if value is None and tag in ctx:
                value = self._as_float(ctx.get(tag), default=None)
            if value is not None:
                signals[tag] = value
        return signals

    def _parse_live_item(self, item: dict, fallback_ts: float):
        od = self._as_float(item.get(OD_TAG))
        if od is None:
            self.live_rejected += 1
            return None

        ctx = {
            tag: self._as_float(item.get(tag), default=0.0)
            for tag in CONTEXT_TAGS
        }
        signals = self._live_segment_signals(item, ctx)
        ts = self._timestamp_seconds(item.get("t_stamp"), fallback=fallback_ts)
        return ts, od, signals, ctx

    def start_live(self, url: str):
        if ws_connect is None:
            raise RuntimeError("websockets not available. pip install websockets>=12")
        if self.live_thread and self.live_thread.is_alive():
            return
        self.live_url = url
        self.live_stop.clear()
        self.live_thread = threading.Thread(target=self._run_live_loop, daemon=True)
        self.live_thread.start()

    def stop_live(self):
        self.live_stop.set()

    def _run_live_loop(self):
        asyncio.run(self._live_main())

    async def _live_main(self):
        while not self.live_stop.is_set():
            try:
                async with ws_connect(self.live_url) as ws:
                    while not self.live_stop.is_set():
                        msg = await ws.recv()
                        data = json.loads(msg)
                        items = data.get("samples") or [data]
                        recv_ts = time.time()
                        for i, item in enumerate(items):
                            n = len(items)
                            fallback_ts = recv_ts - (n - 1 - i) * (1.0 / 2400.0)
                            parsed = self._parse_live_item(item, fallback_ts)
                            if parsed is None:
                                continue
                            try:
                                self.live_queue.put_nowait(parsed)
                            except queue.Full:
                                self.live_dropped += 1
            except Exception:
                await asyncio.sleep(0.5)


DATA = DataStore()
