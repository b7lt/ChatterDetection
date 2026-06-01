import os
import threading, asyncio, json, queue, time

import numpy as np
import pandas as pd

try:
    from websockets.asyncio.client import connect as ws_connect
except Exception:
    ws_connect = None

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

from config import SECONDARY_COL_GUESSES, pick
from status_bar import status
from models import CONTEXT_TAGS, N_CONTEXT


class DataStore:
    def __init__(self):
        self.path = None
        self.ts = []       # list[str] raw ts strings
        self.ts_dt = []    # list[pd.timestamp] parsed timestamps aligned with self.od
        self.od = []       # list[float]
        self.od_hist = []  # ~1 Hz downsampled, never trimmed; used for history display
        self.ts_hist = []  # pd.Timestamp per od_hist sample
        self.classes = []  # list[dict]: {"start":ts, "end":ts, "label":str, "i0":int, "i1":int}
                # secondary / comparison series
        self.paired_df = None # pandas dataframe with columns ["od","sec","t"]

        # Context signal storage — parallel to self.od
        # Keys = CONTEXT_TAGS; populated from WebSocket feed or XLSX sheets.
        self.context_data: dict[str, list[float]] = {tag: [] for tag in CONTEXT_TAGS}

        self.model = None
        self.window_size = None
        self._classified_with_model = None  # tracks which model last classified

        self.available_sheets = []

        self.live_url = "ws://localhost:6467"
        self.live_thread = None
        self.live_stop = threading.Event()
        self.live_queue = queue.Queue(maxsize=10000)

        self._decim_current_sec = None
        self._decim_vals = []
        self._decim_ctx_vals: dict[str, list] = {}   # context accumulator for decimated path
        self._decim_speed_ok = False
        self._hist_current_sec = None
        self._hist_vals = []
        self.target_hz = 1
        self.decimate_enabled = False

        self._trim_offset = 0
        self._MAX_LIVE = 500_000
        self._TRIM_TO  = 400_000

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _read_any_table(self, path: str, sheet=None):
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            if sheet is None:
                return pd.read_excel(path, sheet_name=None, parse_dates=[0])
            else:
                # Always read OD + speed; also read any context sheets present in the file
                xl = pd.ExcelFile(path)
                available = set(xl.sheet_names)
                sheets_to_read = [sheet, "YS_Pullout1_Act_Speed_fpm"]
                for tag in CONTEXT_TAGS:
                    if tag in available:
                        sheets_to_read.append(tag)
                # Deduplicate while preserving order
                seen = set()
                unique_sheets = [s for s in sheets_to_read
                                 if not (s in seen or seen.add(s))]
                return pd.read_excel(path, sheet_name=unique_sheets, parse_dates=[0])

    def _smart_to_numeric(self, series: pd.Series) -> pd.Series:
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

    def filter_by_speed(self, df_dict):
        speed_df = df_dict["YS_Pullout1_Act_Speed_fpm"]
        mask     = speed_df["Tag_value"] > 1
        valid_indices = speed_df[mask].index
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
            excel_file = pd.ExcelFile(path)
            self.available_sheets = excel_file.sheet_names
        else:
            self.available_sheets = []

        df_dict = self._read_any_table(path, "NDC_System_OD_Value")
        if df_dict is None or len(df_dict) == 0:
            raise ValueError("Empty file.")

        filtered_df_dict = self.filter_by_speed(df_dict)

        self.od    = filtered_df_dict["NDC_System_OD_Value"]["Tag_value"].tolist()
        self.ts    = filtered_df_dict["NDC_System_OD_Value"]["t_stamp"].astype(str).tolist()
        self.ts_dt = pd.to_datetime(
            filtered_df_dict["NDC_System_OD_Value"]["t_stamp"], errors="coerce"
        ).tolist()
        self.classes      = []
        self._trim_offset = 0

        # ── Populate context_data from whichever context sheets are present ──
        self.context_data = {tag: [] for tag in CONTEXT_TAGS}
        n_od = len(self.od)
        for tag in CONTEXT_TAGS:
            if tag in filtered_df_dict:
                df_ctx = filtered_df_dict[tag]
                vals = (df_ctx["Tag_value"].tolist()
                        if "Tag_value" in df_ctx.columns else [])
                # Pad or trim to match OD length
                if len(vals) < n_od:
                    vals = vals + [0.0] * (n_od - len(vals))
                self.context_data[tag] = vals[:n_od]
            else:
                self.context_data[tag] = [0.0] * n_od

        # ── Build 1 Hz historical buffer ─────────────────────────────────────
        self.od_hist = []
        self.ts_hist = []
        self._hist_current_sec = None
        self._hist_vals = []
        try:
            df_h = pd.DataFrame({"t": self.ts_dt, "od": self.od})
            df_h["t"] = pd.to_datetime(df_h["t"])
            df_h = df_h.set_index("t").resample("1s").median().dropna().reset_index()
            self.od_hist = df_h["od"].tolist()
            self.ts_hist = df_h["t"].tolist()
        except Exception:
            pass

        try:
            v = self.od
            status(f"Loaded: rows={len(v)}  min={min(v):.6g}  max={max(v):.6g}  mean={sum(v)/len(v):.6g}")
        except Exception:
            pass

        if self.model is not None and app is not None:
            self.auto_classify(window_size=self.window_size)

    # ── Secondary / correlation helpers ───────────────────────────────────────

    def _align_series(self, df_main, tcol_main, ycol_main, df_sec, tcol_sec, ycol_sec):
        m = pd.DataFrame({
            "t":  pd.to_datetime(df_main[tcol_main], errors="coerce"),
            "od": self._smart_to_numeric(df_main[ycol_main]),
        }).dropna()
        s = pd.DataFrame({
            "t":   pd.to_datetime(df_sec[tcol_sec], errors="coerce"),
            "sec": self._smart_to_numeric(df_sec[ycol_sec]),
        }).dropna()

        for col in ["t"]:
            for df in [m, s]:
                try:   df[col] = df[col].dt.tz_convert(None)
                except: df[col] = df[col].dt.tz_localize(None)

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

        df_dict  = pd.read_excel(self.path, sheet_name=[sheet_name, "YS_Pullout1_Act_Speed_fpm"])
        df_sec   = df_dict[sheet_name]
        if df_sec is None or df_sec.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")

        filtered_dict    = self.filter_by_speed(df_dict)
        df_sec_filtered  = filtered_dict[sheet_name]
        cols_s           = list(df_sec_filtered.columns)
        tcol_s           = pick(cols_s, SECONDARY_COL_GUESSES["time"]) or "t_stamp"
        ycol_s           = pick(cols_s, SECONDARY_COL_GUESSES["val"])  or "Tag_value"

        if tcol_s not in cols_s:
            raise ValueError(f"Could not find time column in '{sheet_name}'. Columns: {cols_s}")
        if ycol_s not in cols_s:
            raise ValueError(f"Could not find value column in '{sheet_name}'. Columns: {cols_s}")

        df_dict_od       = pd.read_excel(self.path,
                                          sheet_name=["NDC_System_OD_Value",
                                                      "YS_Pullout1_Act_Speed_fpm"])
        filtered_dict_od = self.filter_by_speed(df_dict_od)
        df_main_filtered = filtered_dict_od["NDC_System_OD_Value"]

        paired = self._align_series(df_main_filtered, "t_stamp", "Tag_value",
                                    df_sec_filtered,  tcol_s,    ycol_s)
        if paired.empty:
            raise ValueError(
                f"No overlapping timestamps between OD and '{sheet_name}' after speed filtering.")

        self.paired_df = paired
        status(f"Secondary loaded & aligned: {sheet_name} • paired rows={len(self.paired_df)}")

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
        K = min(max_lag_samples, n - 2)
        for k in range(1, K + 1):
            r_pos = float(np.corrcoef(x[k:], y[:-k])[0, 1])
            if r_pos > best_r: best_r, best_k = r_pos, +k
            r_neg = float(np.corrcoef(x[:-k], y[k:])[0, 1])
            if r_neg > best_r: best_r, best_k = r_neg, -k
        return {"n": n, "pearson_r": r0, "best_lag": best_k, "r_at_best_lag": best_r}

    def _paired_ok(self):
        return (self.paired_df is not None) and (not self.paired_df.empty)

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
            else:       r[i] = np.corrcoef(x,       y)[0, 1]
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
            i1   = i0 + win_samples
            segx, segy = x[i0:i1], y[i0:i1]
            r    = (float(np.corrcoef(segx, segy)[0, 1])
                    if np.std(segx) > 1e-12 and np.std(segy) > 1e-12
                    else np.nan)
            mids.append(t[i0 + win_samples // 2])
            rr.append(r)
        return np.array(mids), np.array(rr)

    # ── Classification helpers ─────────────────────────────────────────────────

    def current_class(self):
        if not self.classes:
            return None, None
        return self.classes[-1]["label"], self.classes[-1]["risk"]

    def _is_hybrid_model(self) -> bool:
        """True when the loaded model is a HybridChatterNet (has a context MLP branch)."""
        return getattr(self.model, "is_hybrid", False) or hasattr(self.model, "mlp")

    def _build_context_vector(self, local_start: int, local_end: int) -> list[float]:
        """
        Per-window context feature vector: mean of each CONTEXT_TAG over [local_start, local_end).
        Returns a list of length N_CONTEXT; missing tags are filled with 0.
        """
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

    def _cnn_infer(self, windows_list, context_list=None):
        """
        Run the loaded model over a list of raw OD windows.
        Each OD window is z-score normalised per-sample (matches training).

        For HybridChatterNet (detected via is_hybrid attribute):
          - context_list must be a list of float vectors, one per window
          - stored ctx_mean / ctx_std are applied automatically

        Returns ndarray shape (N, 2) for HybridChatterNet
                or (N, num_classes) for legacy ChatterCNN.
        Column index 1 = chatter probability in both cases.
        """
        import torch

        self.model.eval()

        # ── Prepare OD tensor ────────────────────────────────────────────────
        segs = []
        for w in windows_list:
            seg = np.asarray(w, dtype=np.float32)
            mu, sigma = seg.mean(), seg.std()
            seg = (seg - mu) / (sigma + 1e-8)
            segs.append(seg)
        X = torch.tensor(np.stack(segs), dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            if self._is_hybrid_model() and context_list is not None:
                # ── Hybrid path ──────────────────────────────────────────────
                ctx = np.array(context_list, dtype=np.float32)   # (N, N_CONTEXT)
                # Apply stored normalizer
                ctx_mean = np.asarray(self.model.ctx_mean, dtype=np.float32)
                ctx_std  = np.asarray(self.model.ctx_std,  dtype=np.float32)
                ctx = (ctx - ctx_mean) / (ctx_std + 1e-8)
                X_ctx = torch.tensor(ctx, dtype=torch.float32)
                return self.model(X, X_ctx).numpy()
            else:
                # ── Pure CNN path (backward-compatible) ──────────────────────
                return self.model(X).numpy()

    def get_label_from_risk_prob(self, risk):
        if   risk < 0.40: return "No Chatter"
        elif risk < 0.70: return "Mild Chatter"
        else:             return "Heavy Chatter"

    def auto_classify(self, window_size=60):
        if self.model is None:
            status("No model selected. Please select a model first.")
            return
        if window_size is None or window_size <= 0:
            status("Invalid window size.")
            return
        if len(self.od) < window_size:
            return

        total_abs  = len(self.od) + self._trim_offset
        num_windows = total_abs // window_size

        if self.classes and (
            self.classes[0]["i1"] - self.classes[0]["i0"] != window_size
            or self._classified_with_model is not self.model
        ):
            self.classes = []

        already_classified = len(self.classes)
        if already_classified >= num_windows:
            return

        windows      = []
        context_list = []
        window_meta  = []

        use_hybrid = self._is_hybrid_model()

        for i in range(already_classified, num_windows):
            abs_start   = i * window_size
            abs_end     = abs_start + window_size
            local_start = abs_start - self._trim_offset
            local_end   = abs_end   - self._trim_offset
            if local_start < 0 or local_end > len(self.od):
                continue
            windows.append(self.od[local_start:local_end])
            if use_hybrid:
                context_list.append(self._build_context_vector(local_start, local_end))
            window_meta.append((abs_start, abs_end, local_start, local_end))

        if not windows:
            return

        probas = self._cnn_infer(
            windows,
            context_list if (use_hybrid and context_list) else None
        )

        for i, (abs_start, abs_end, local_start, local_end) in enumerate(window_meta):
            chatter_confidence = float(probas[i][1])
            self.classes.append({
                "start": self.ts[local_start],
                "end":   self.ts[local_end - 1],
                "label": self.get_label_from_risk_prob(chatter_confidence),
                "i0":    abs_start,
                "i1":    abs_end,
                "risk":  chatter_confidence,
            })

        self._classified_with_model = self.model
        status(f"Auto-classes computed: {len(self.classes)}")

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _linreg_slope(y):
        n = len(y)
        if n < 2: return 0.0
        sx   = n * (n - 1) / 2.0
        sxx  = n * (n - 1) * (2 * n - 1) / 6.0
        sy   = sum(y)
        sxy  = sum(i * yi for i, yi in enumerate(y))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12: return 0.0
        return (n * sxy - sx * sy) / denom

    def recent_window(self, n=1024):
        if not self.od: return []
        return self.od[-min(n, len(self.od)):]

    def trend_slope(self, n=1024):
        y = self.recent_window(n)
        return self._linreg_slope(y) if y else 0.0

    # ── Sample append (keeps context in sync) ─────────────────────────────────

    def _append_sample(self, ts_dt, od, context: dict | None = None):
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
                if len(lst) > excess:
                    del lst[:excess]
            self._trim_offset += excess

    # ── Live queue consumer ────────────────────────────────────────────────────

    def _consume_live_queue(self):
        drained = 0
        while True:
            try:
                item = self.live_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1

            ts, od, speed, ctx = item   # ctx: dict of context tag → float

            # ── Always maintain 1 Hz historical buffer ────────────────────────
            hist_sec = int(ts)
            if self._hist_current_sec is None:
                self._hist_current_sec = hist_sec
            if hist_sec != self._hist_current_sec:
                if self._hist_vals:
                    self.od_hist.append(float(np.median(self._hist_vals)))
                    self.ts_hist.append(pd.to_datetime(self._hist_current_sec, unit="s"))
                self._hist_current_sec = hist_sec
                self._hist_vals = []
            if speed is not None and speed > 1:
                self._hist_vals.append(od)

            # ── Non-decimated path ────────────────────────────────────────────
            if not getattr(self, "decimate_enabled", False):
                if speed is None or speed <= 1:
                    continue
                self._append_sample(pd.to_datetime(ts, unit="s"), od, ctx)
                self._decim_current_sec = None
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False
                continue

            # ── Decimated path (≈1 Hz median) ─────────────────────────────────
            sec = int(ts)
            if self._decim_current_sec is None:
                self._decim_current_sec = sec
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False

            if sec != self._decim_current_sec:
                # Flush previous second
                if self._decim_speed_ok and self._decim_vals:
                    ctx_median = {
                        tag: float(np.median(vals)) if vals else 0.0
                        for tag, vals in self._decim_ctx_vals.items()
                    }
                    self._append_sample(
                        pd.to_datetime(self._decim_current_sec, unit="s"),
                        float(np.median(self._decim_vals)),
                        ctx_median,
                    )
                self._decim_current_sec = sec
                self._decim_vals        = []
                self._decim_ctx_vals    = {}
                self._decim_speed_ok    = False

            if speed is not None and speed > 1:
                self._decim_speed_ok = True
                self._decim_vals.append(od)
                for tag, val in ctx.items():
                    self._decim_ctx_vals.setdefault(tag, []).append(val)

        return drained

    # ── WebSocket live feed ────────────────────────────────────────────────────

    def start_live(self, url: str):
        if ws_connect is None:
            raise RuntimeError("websockets is not available. Install `websockets` >= 12.")
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
                        msg  = await ws.recv()
                        data = json.loads(msg)
                        items    = data.get("samples") or [data]
                        recv_ts  = time.time()
                        for i, item in enumerate(items):
                            od    = float(item.get("NDC_System_OD_Value", "nan"))
                            speed = item.get("YS_Pullout1_Act_Speed_fpm", None)
                            # Extract all context signals from the message
                            ctx = {tag: float(item.get(tag, 0.0)) for tag in CONTEXT_TAGS}
                            # Back-calculate timestamps for batch members
                            n  = len(items)
                            ts = recv_ts - (n - 1 - i) * (1.0 / 2400.0)
                            try:
                                self.live_queue.put_nowait((ts, od, speed, ctx))
                            except queue.Full:
                                pass
            except Exception:
                await asyncio.sleep(0.5)


DATA = DataStore()