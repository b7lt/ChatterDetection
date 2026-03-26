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

        self.model = None
        self.window_size = None

        self.available_sheets = []

        self.live_url = "ws://localhost:6467"
        self.live_thread = None
        self.live_stop = threading.Event()
        self.live_queue = queue.Queue(maxsize=10000)

        self._decim_current_sec = None
        self._decim_vals = []
        self._decim_speed_ok = False
        self._hist_current_sec = None  # 1Hz accumulator for od_hist
        self._hist_vals = []
        self.target_hz = 1
        # whether to decimate high-rate live data to 1 Hz (median per second)
        self.decimate_enabled = False

        # Live-buffer cap: keep at most _MAX_LIVE samples; older samples are
        # dropped from the front and counted in _trim_offset so that class
        # i0/i1 indices (stored as absolute sample numbers) stay correct.
        self._trim_offset = 0
        self._MAX_LIVE = 500_000   # samples kept in memory
        self._TRIM_TO  = 400_000   # trim back to this size when cap is hit

    def _read_any_table(self, path: str, sheet=None):
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            if sheet is None:
                return pd.read_excel(path, sheet_name=None, parse_dates=[0])
            else:
                return pd.read_excel(path, sheet_name=[sheet, 'YS_Pullout1_Act_Speed_fpm'], parse_dates=[0])

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
        speed_df = df_dict['YS_Pullout1_Act_Speed_fpm']

        speed_threshold = 1

        mask = speed_df['Tag_value'] > speed_threshold
        filtered_speed = speed_df[mask]
        valid_indices = filtered_speed.index

        filtered_dict = {}
        for sheet_name, df in df_dict.items():
            if len(df) == 0:
                continue

            filtered_dict[sheet_name] = df.reindex(valid_indices).reset_index(drop=True)

        return filtered_dict

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

        self.od = filtered_df_dict['NDC_System_OD_Value']['Tag_value'].tolist()
        self.ts = filtered_df_dict['NDC_System_OD_Value']['t_stamp'].astype(str).tolist()
        self.ts_dt = pd.to_datetime(filtered_df_dict['NDC_System_OD_Value']['t_stamp'], errors='coerce').tolist()
        self.classes = []
        self._trim_offset = 0

        # Build 1Hz historical buffer from loaded data
        self.od_hist = []
        self.ts_hist = []
        self._hist_current_sec = None
        self._hist_vals = []
        try:
            df_h = pd.DataFrame({'t': self.ts_dt, 'od': self.od})
            df_h['t'] = pd.to_datetime(df_h['t'])
            df_h = df_h.set_index('t').resample('1s').median().dropna().reset_index()
            self.od_hist = df_h['od'].tolist()
            self.ts_hist = df_h['t'].tolist()
        except Exception:
            pass

        try:
            v = self.od
            status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}' • rows={len(v)} "
                   f"• min={v.min():.6g}, max={v.max():.6g}, mean={v.mean():.6g}")
        except Exception:
            pass

        self.path = path

        try:
            status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}'")
        except Exception:
            pass

        if self.model is not None and app is not None:
            self.auto_classify(window_size=self.window_size)

    def _align_series(self, df_main, tcol_main, ycol_main, df_sec, tcol_sec, ycol_sec):
        # prepare main
        m = pd.DataFrame({
            "t": pd.to_datetime(df_main[tcol_main], errors="coerce"),
            "od": self._smart_to_numeric(df_main[ycol_main]),
        }).dropna()
        # prepare secondary
        s = pd.DataFrame({
            "t": pd.to_datetime(df_sec[tcol_sec], errors="coerce"),
            "sec": self._smart_to_numeric(df_sec[ycol_sec]),
        }).dropna()

        # remove timezone
        for col in ["t"]:
            if hasattr(m[col], "dt"):
                try: m[col] = m[col].dt.tz_convert(None)
                except: m[col] = m[col].dt.tz_localize(None)
            if hasattr(s[col], "dt"):
                try: s[col] = s[col].dt.tz_convert(None)
                except: s[col] = s[col].dt.tz_localize(None)

        m = m.sort_values("t")
        s = s.sort_values("t")

        tol = pd.Timedelta(seconds=1)
        if len(m) >= 3:
            dtm = (m["t"].diff().dropna().median() or pd.Timedelta(seconds=1))
            tol = max(tol, dtm)
        if len(s) >= 3:
            dts = (s["t"].diff().dropna().median() or pd.Timedelta(seconds=1))
            tol = max(tol, dts)

        paired = pd.merge_asof(m, s, on="t", direction="nearest", tolerance=tol)
        paired = paired.dropna().reset_index(drop=True)
        return paired[["t", "od", "sec"]]

    def load_secondary_sheet(self, sheet_name: str):
        if not self.path:
            raise ValueError("Load the main data file first.")
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)

        ext = os.path.splitext(self.path.lower())[1]
        if ext not in [".xlsx", ".xls"]:
            raise ValueError("Secondary sheet loading only works with Excel files.")

        df_dict = pd.read_excel(self.path, sheet_name=[sheet_name, 'YS_Pullout1_Act_Speed_fpm'])

        df_sec = df_dict[sheet_name]
        if df_sec is None or df_sec.empty:
            raise ValueError(f"Sheet '{sheet_name}' is empty.")

        filtered_dict = self.filter_by_speed(df_dict)
        df_sec_filtered = filtered_dict[sheet_name]

        cols_s = list(df_sec_filtered.columns)
        tcol_s = pick(cols_s, SECONDARY_COL_GUESSES["time"]) or "t_stamp"
        ycol_s = pick(cols_s, SECONDARY_COL_GUESSES["val"]) or "Tag_value"

        if tcol_s not in cols_s:
            raise ValueError(f"Could not find time column in sheet '{sheet_name}'. Columns: {cols_s}")
        if ycol_s not in cols_s:
            raise ValueError(f"Could not find value column in sheet '{sheet_name}'. Columns: {cols_s}")

        df_dict_od = pd.read_excel(self.path, sheet_name=['NDC_System_OD_Value', 'YS_Pullout1_Act_Speed_fpm'])
        filtered_dict_od = self.filter_by_speed(df_dict_od)
        df_main_filtered = filtered_dict_od['NDC_System_OD_Value']

        tcol_m = "t_stamp"
        ycol_m = "Tag_value"

        paired = self._align_series(df_main_filtered, tcol_m, ycol_m,
                                    df_sec_filtered, tcol_s, ycol_s)
        if paired.empty:
            raise ValueError(f"No overlapping timestamps between OD and '{sheet_name}' after speed filtering.")

        self.paired_df = paired

        status(f"Secondary loaded & aligned: {sheet_name} • paired rows={len(self.paired_df)} (speed filtered)")

    def corr_stats(self, max_lag_samples: int = 300):
        """
        Compute basic correlation stats between OD and secondary series.
        Returns dict with: n, pearson_r, best_lag, r_at_best_lag.
        """
        if self.paired_df is None or self.paired_df.empty:
            return {"n": 0, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}

        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 3:
            return {"n": n, "pearson_r": np.nan, "best_lag": 0, "r_at_best_lag": np.nan}

        # plain Pearson (no lag)
        r0 = float(np.corrcoef(x, y)[0, 1])

        # best lag (shift y relative to x)
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
        """
        Return lags (in samples) and Pearson r at each lag (y shifted).
        lags: array from -K..+K, r: same length.
        """
        if not self._paired_ok(): return np.array([]), np.array([])
        x = self.paired_df["od"].to_numpy(dtype=float)
        y = self.paired_df["sec"].to_numpy(dtype=float)
        n = min(len(x), len(y))
        if n < 5: return np.array([]), np.array([])
        K = int(min(max_lag_samples, n - 3))
        lags = np.arange(-K, K + 1, dtype=int)
        r = np.zeros_like(lags, dtype=float)
        for i, k in enumerate(lags):
            if k < 0:   # y leads (shift y forward)
                r[i] = np.corrcoef(x[:k], y[-k:])[0, 1]
            elif k > 0: # x leads
                r[i] = np.corrcoef(x[k:], y[:-k])[0, 1]
            else:
                r[i] = np.corrcoef(x, y)[0, 1]
        return lags, r

    def rolling_corr(self, win_samples=200, step=10):
        """
        Rolling Pearson r over paired series.
        Returns arrays (t_mid, r_roll). t_mid are midpoint timestamps of each window.
        """
        if not self._paired_ok(): return np.array([]), np.array([])
        df = self.paired_df
        x = df["od"].to_numpy(dtype=float)
        y = df["sec"].to_numpy(dtype=float)
        t = pd.to_datetime(df["t"], errors="coerce").to_numpy()
        n = len(df)
        if n < max(10, win_samples): return np.array([]), np.array([])
        mids, rr = [], []
        for i0 in range(0, n - win_samples + 1, step):
            i1 = i0 + win_samples
            segx, segy = x[i0:i1], y[i0:i1]
            if np.std(segx) < 1e-12 or np.std(segy) < 1e-12:
                r = np.nan
            else:
                r = float(np.corrcoef(segx, segy)[0, 1])
            mids.append(t[i0 + win_samples // 2])
            rr.append(r)
        return np.array(mids), np.array(rr)

    def current_class(self):
        if not self.classes:
            return None, None
        return self.classes[-1]["label"], self.classes[-1]["risk"]

    def _cnn_infer(self, windows_list):
        """
        Run the loaded CNN over a list of raw OD windows.
        Each window is z-score normalised per-sample (matches training).
        Returns ndarray shape (N, num_classes); column 1 = chatter probability.
        """
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

        # i0/i1 are stored as absolute sample indices (never reset by trimming).
        # Convert to local buffer indices via self._trim_offset.
        total_abs = len(self.od) + self._trim_offset
        num_windows = total_abs // window_size

        # If window_size changed, start fresh
        if self.classes and self.classes[0]["i1"] - self.classes[0]["i0"] != window_size:
            self.classes = []

        already_classified = len(self.classes)
        if already_classified >= num_windows:
            return  # nothing new to classify

        windows = []
        window_metadata = []
        for i in range(already_classified, num_windows):
            abs_start = i * window_size
            abs_end   = abs_start + window_size
            local_start = abs_start - self._trim_offset
            local_end   = abs_end   - self._trim_offset
            if local_start < 0 or local_end > len(self.od):
                continue  # data trimmed or incomplete window
            windows.append(self.od[local_start:local_end])
            window_metadata.append((abs_start, abs_end, local_start, local_end))

        if not windows:
            return

        probas = self._cnn_infer(windows)

        for i, (abs_start, abs_end, local_start, local_end) in enumerate(window_metadata):
            chatter_confidence = float(probas[i][1])  # class 1 = chatter
            self.classes.append({
                "start": self.ts[local_start],
                "end": self.ts[local_end - 1],
                "label": self.get_label_from_risk_prob(chatter_confidence),
                "i0": abs_start,   # absolute index
                "i1": abs_end,     # absolute index
                "risk": chatter_confidence
            })

        status(f"Auto-classes computed: {len(self.classes)}")

    # ---- math helpers (no numpy) ----
    @staticmethod
    def _linreg_slope(y):
        """Least-squares slope over y with x = 0..n-1 (pure Python)."""
        n = len(y)
        if n < 2: return 0.0
        sx = n * (n - 1) / 2.0
        sxx = n * (n - 1) * (2 * n - 1) / 6.0
        sy = sum(y)
        sxy = sum(i * yi for i, yi in enumerate(y))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12: return 0.0
        return (n * sxy - sx * sy) / denom

    def recent_window(self, n=1024):
        if not self.od: return []
        n = min(n, len(self.od))
        return self.od[-n:]

    def trend_slope(self, n=1024):
        y = self.recent_window(n)
        return self._linreg_slope(y) if y else 0.0

    def _append_sample(self, ts_dt, od):
        self.ts_dt.append(ts_dt)
        self.ts.append(str(ts_dt))
        self.od.append(float(od))
        # Drop oldest samples in a batch when the cap is hit so list-delete
        # cost is amortised (one O(n) shift every _MAX_LIVE - _TRIM_TO appends).
        if len(self.od) > self._MAX_LIVE:
            excess = len(self.od) - self._TRIM_TO
            del self.od[:excess]
            del self.ts[:excess]
            del self.ts_dt[:excess]
            self._trim_offset += excess

    def _consume_live_queue(self):
        """Called from the UI thread (poll) to drain queue. If decimation is enabled,
        bucket samples into 1-second windows and append the median (≈1 Hz). If disabled,
        append every sample at its native rate for high-resolution analysis (e.g., FFT)."""
        drained = 0
        while True:
            try:
                item = self.live_queue.get_nowait()
            except queue.Empty:
                break
            drained += 1
            ts, od, speed = item  # ts is a float seconds since epoch

            # ---- ALWAYS maintain 1Hz historical buffer (never trimmed) ----
            hist_sec = int(ts)
            if self._hist_current_sec is None:
                self._hist_current_sec = hist_sec
            if hist_sec != self._hist_current_sec:
                if self._hist_vals:
                    self.od_hist.append(float(np.median(self._hist_vals)))
                    self.ts_hist.append(pd.to_datetime(self._hist_current_sec, unit='s'))
                self._hist_current_sec = hist_sec
                self._hist_vals = []
            if speed is not None and speed > 1:
                self._hist_vals.append(od)

            # ---- NON-DECIMATED PATH ----
            if not getattr(self, "decimate_enabled", True):
                # Only keep samples while line is actually moving
                if speed is None or speed <= 1:
                    continue
                self._append_sample(pd.to_datetime(ts, unit="s"), od)
                # Reset any pending decimation bucket so we don't emit a stale median later
                self._decim_current_sec = None
                self._decim_vals = []
                self._decim_speed_ok = False
                continue

            # ---- DECIMATED PATH (≈1 Hz median, only if speed>1 in that second) ----
            sec = int(ts)
            if self._decim_current_sec is None:
                self._decim_current_sec = sec
                self._decim_vals = []
                self._decim_speed_ok = False

            if sec != self._decim_current_sec:
                # flush previous second
                if self._decim_speed_ok and self._decim_vals:
                    median_od = float(np.median(self._decim_vals))
                    self._append_sample(pd.to_datetime(self._decim_current_sec, unit="s"), median_od)
                # start next bucket
                self._decim_current_sec = sec
                self._decim_vals = []
                self._decim_speed_ok = False

            if speed is not None and speed > 1:
                self._decim_speed_ok = True
                self._decim_vals.append(od)

        return drained

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
                        msg = await ws.recv()
                        data = json.loads(msg)
                        # server may send a batch: {"samples": [{...}, ...]}
                        # or a single sample for backwards compatibility
                        items = data.get("samples") or [data]
                        recv_ts = time.time()
                        for i, item in enumerate(items):
                            od = float(item.get("NDC_System_OD_Value", "nan"))
                            speed = item.get("YS_Pullout1_Act_Speed_fpm", None)
                            # back-calculate timestamps so samples are evenly spaced
                            n = len(items)
                            ts = recv_ts - (n - 1 - i) * (1.0 / 2400.0)
                            try:
                                self.live_queue.put_nowait((ts, od, speed))
                            except queue.Full:
                                pass
            except Exception:
                await asyncio.sleep(0.5)


DATA = DataStore()
