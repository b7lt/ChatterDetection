import os, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import numpy as np, pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pickle
from PIL import Image, ImageTk
import threading, asyncio, json, queue, time
try:
    from websockets.asyncio.client import connect as ws_connect
except Exception:
    ws_connect = None

APP_TITLE   = "Wavy Detection Dashboard"
APP_VERSION = "v1.0"

EXACT_OD_COLUMN = "Tag_value"
ENFORCE_EXACT_OD = True

DATA_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    "od":   ["od", "outer_diameter", "tube_od", "ndc_od_value","ndc_system_ovality_value__tag_value", "tag_value"],
}

SECONDARY_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    "val": [
        "ovality", "ovality_value",
        "ndc_system_ovality_value", "ndc_system_ovality_value__tag_value",
        "tag_value", "value"
    ],
}

CLASS_COLORS = {
    "STEADY":        "#4CAF50",
    "MILD_WAVE":     "#FFB300",
    "STRONG_WAVE":   "#E53935",
    "DRIFT":         "#2196F3",
    "BURSTY_NOISY":  "#8E24AA",
    "UNCERTAIN":     "#9E9E9E",
}

def pastel(hex_color: str, alpha: float = 0.25) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # blend toward white
    r = int((1 - alpha) * 255 + alpha * r)
    g = int((1 - alpha) * 255 + alpha * g)
    b = int((1 - alpha) * 255 + alpha * b)
    return f"#{r:02X}{g:02X}{b:02X}"


def normalize_label(s) -> str:
    if s is None:
        return "UNCERTAIN"

    try:
        v = int(float(str(s).strip()))
        id_map = {0: "STEADY", 1: "MILD_WAVE", 2: "STRONG_WAVE", 3: "DRIFT", 4: "BURSTY_NOISY", 5: "UNCERTAIN"}
        return id_map.get(v, "UNCERTAIN")
    except Exception:
        pass

    t = str(s).strip().upper()
    t = t.replace("-", "_").replace(" ", "_").replace("/", "_")

    aliases = {
        # steady / good
        "STEADY": "STEADY", "OK": "STEADY", "GOOD": "STEADY", "NORMAL": "STEADY", "FLAT": "STEADY",
        "STABLE": "STEADY", "STEADY_STATE": "STEADY",

        # mild wave
        "MILD": "MILD_WAVE", "MILD_WAVE": "MILD_WAVE", "LIGHT_WAVE": "MILD_WAVE",
        "SMALL_WAVE": "MILD_WAVE", "MINOR_WAVE": "MILD_WAVE",

        # strong wave / chatter
        "STRONG": "STRONG_WAVE", "STRONG_WAVE": "STRONG_WAVE",
        "WAVY": "STRONG_WAVE", "CHATTER": "STRONG_WAVE", "HEAVY_WAVE": "STRONG_WAVE",

        # drift / trend
        "DRIFT": "DRIFT", "DRIFTING": "DRIFT", "TREND": "DRIFT", "SLOPE": "DRIFT",

        # noisy / burst
        "BURSTY_NOISY": "BURSTY_NOISY", "BURSTY_NOISE": "BURSTY_NOISY",
        "NOISY": "BURSTY_NOISY", "BURST": "BURSTY_NOISY", "SPIKY": "BURSTY_NOISY",

        # unknown
        "UNCERTAIN": "UNCERTAIN", "UNKNOWN": "UNCERTAIN",
    }
    return aliases.get(t, "UNCERTAIN")

VISIBLE_CLASSES = set(CLASS_COLORS.keys())

def pick(colnames, candidates):
    low = [c.lower() for c in colnames]
    for alias in candidates:
        if alias.lower() in low:
            return colnames[low.index(alias.lower())]
    return None

class DataStore:
    def __init__(self):
        self.path = None
        self.ts = []       # list[str] raw ts strings
        self.ts_dt = []    # list[pd.timestamp] parsed timestamps aligned with self.od
        self.od = []       # list[float]
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
        self.target_hz = 1
        # whether to decimate high-rate live data to 1 Hz (median per second)
        self.decimate_enabled = False

    def _read_any_table(self, path: str, sheet = None):
        ext = os.path.splitext(path.lower())[1]
        if ext in [".xlsx", ".xls"]:
            if sheet == None:
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

        try:
            v = self.od
            App.status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}' • rows={len(v)} "
                    f"• min={v.min():.6g}, max={v.max():.6g}, mean={v.mean():.6g}")
        except Exception:
            pass

        self.path = path

        try:
            App.status(f"Using time='{'t_stamp'}', OD='{'Tag_value'}'")
        except Exception:
            pass

        if self.model is not None and app is not None:
            self.auto_classify(window_size=self.window_size)
            if hasattr(app, 'pages') and 'Analysis' in app.pages:
                analysis_page = app.pages['Analysis']
                analysis_page.update_confidence_timeline()

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
        return paired[["t","od","sec"]]
    
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

        App.status(f"Secondary loaded & aligned: {sheet_name} • paired rows={len(self.paired_df)} (speed filtered)")

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
        r0 = float(np.corrcoef(x, y)[0,1])

        # best lag (shift y relative to x)
        best_r, best_k = r0, 0
        K = min(max_lag_samples, n-2)
        for k in range(1, K+1):
            r_pos = float(np.corrcoef(x[k:], y[:-k])[0,1])
            if r_pos > best_r: best_r, best_k = r_pos, +k
            r_neg = float(np.corrcoef(x[:-k], y[k:])[0,1])
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
        lags = np.arange(-K, K+1, dtype=int)
        r = np.zeros_like(lags, dtype=float)
        for i, k in enumerate(lags):
            if k < 0:   # y leads (shift y forward)
                r[i] = np.corrcoef(x[:k], y[-k:])[0,1]
            elif k > 0: # x leads
                r[i] = np.corrcoef(x[k:], y[:-k])[0,1]
            else:
                r[i] = np.corrcoef(x, y)[0,1]
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
                r = float(np.corrcoef(segx, segy)[0,1])
            mids.append(t[i0 + win_samples//2])
            rr.append(r)
        return np.array(mids), np.array(rr)

    def current_class(self):
        if not self.classes:
            return None, None
        return self.classes[-1]["label"], self.classes[-1]["risk"]

    def extract_features(self, window_data):
        window_data = np.asarray(window_data, dtype=float)
        mean_val = float(np.mean(window_data))
        std_val = float(np.std(window_data))
        ptp_val = float(np.ptp(window_data))

        safe_mean = mean_val if mean_val != 0 else 1e-10

        features = {
            # absolute scale
            "mean_od": mean_val,
            "std_abs": std_val,
            "range_abs": ptp_val,
            # relative-to-mean scale
            "coef_variation": std_val / safe_mean,
            "relative_range": ptp_val / safe_mean,
            "normalized_variance": float(np.var(window_data)) / (safe_mean ** 2),
            "peak_to_peak_ratio": ptp_val / abs(safe_mean),
            "relative_max_deviation": (float(np.max(window_data)) - mean_val) / safe_mean,
            "relative_min_deviation": (mean_val - float(np.min(window_data))) / safe_mean,
            "mean_abs_diff": float(np.mean(np.abs(np.diff(window_data)))) / safe_mean,
            "max_abs_diff": float(np.max(np.abs(np.diff(window_data)))) / safe_mean,
        }

        # estimate sampling frequ
        fs = 2400.0
        try:
            if self.ts_dt and len(self.ts_dt) >= n:
                t = pd.to_datetime(self.ts_dt[-n:], errors="coerce")
                t = t[~pd.isna(t)]
                if len(t) >= 2:
                    dt_sec = np.median(
                        np.diff(t).astype("timedelta64[ns]").astype(np.float64)
                    ) / 1e9
                    if dt_sec > 0:
                        fs = float(1.0 / dt_sec)
        except Exception:
            pass

        # FFT features
        fft_valid = 0
        fft_peak_freq = 0.0
        fft_peak_prominence = 0.0

        y = window_data - mean_val
        if len(y) >= 16 and not np.allclose(y, 0.0, atol=1e-12):
            nfft = len(y)
            freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
            fft_vals = np.fft.rfft(y)
            psd = (np.abs(fft_vals) ** 2) / nfft

            if psd.size > 1 and np.any(np.isfinite(psd)):
                psd_no_dc = psd[1:]
                freqs_no_dc = freqs[1:]
                total_power = float(np.sum(psd_no_dc))

                if total_power > 0:
                    peak_idx = int(np.argmax(psd_no_dc))
                    peak_power = float(psd_no_dc[peak_idx])
                    fft_peak_freq = float(freqs_no_dc[peak_idx])
                    fft_peak_prominence = peak_power / total_power

                    if fft_peak_prominence > 0.30:
                        fft_valid = 1

        features["fft_valid"] = fft_valid
        features["fft_peak_freq"] = fft_peak_freq
        features["fft_peak_prominence"] = fft_peak_prominence

        return features
    
    def get_label_from_risk_prob(self, risk):
        if   risk < 0.25: return "STEADY"
        elif risk < 0.45: return "MILD_WAVE"
        elif risk < 0.65: return "DRIFT"
        elif risk < 0.80: return "BURSTY_NOISY"
        else:             return "STRONG_WAVE"

    def auto_classify(self, window_size=60):
        if self.model is None:
            App.status("No model selected. Please select a model first.")
            return
        if window_size is None or window_size <= 0:
            App.status("Invalid window size.")
            return
        if len(self.od) < window_size:
            return
        
        # clear existing classes
        self.classes = []
        
        num_windows = len(self.od) // window_size

        # collect all features first
        features_list = []
        window_metadata = []  # store start/end indices for each window
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window = self.od[start_idx:end_idx]
            
            # extract features as a dict
            features_dict = self.extract_features(window)
            features_list.append(features_dict)
            window_metadata.append((start_idx, end_idx))
        
        # convert to DataFrame (same as training)
        X = pd.DataFrame(features_list)
        
        # scale all features at once (same as training)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        
        # predict for all windows
        if not features_list:
            return
        probas = self.model.predict_proba(X)
        
        # create class segments
        for i, (start_idx, end_idx) in enumerate(window_metadata):
            chatter_confidence = probas[i][1]  # probability of class 1 (chatter/bad)
            
            self.classes.append({
                "start": self.ts[start_idx],
                "end": self.ts[end_idx - 1] if end_idx > 0 else self.ts[0],
                "label": self.get_label_from_risk_prob(chatter_confidence),
                "i0": start_idx,
                "i1": end_idx,
                "risk": chatter_confidence
            })

        App.status(f"Auto-classes computed: {len(self.classes)} span(s).")


    # ---- math helpers (no numpy) ----
    @staticmethod
    def _linreg_slope(y):
        """Least-squares slope over y with x = 0..n-1 (pure Python)."""
        n = len(y)
        if n < 2: return 0.0
        sx = n*(n-1)/2.0
        sxx = n*(n-1)*(2*n-1)/6.0
        sy = sum(y)
        sxy = sum(i*yi for i, yi in enumerate(y))
        denom = n*sxx - sx*sx
        if abs(denom) < 1e-12: return 0.0
        return (n*sxy - sx*sy) / denom

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
                        # server sends {"od": float, "speed": int}
                        od = float(data.get("od", "nan"))
                        speed = data.get("speed", None)
                        ts = time.time()
                        try:
                            self.live_queue.put_nowait((ts, od, speed))
                        except queue.Full:
                            pass
            except Exception:
                await asyncio.sleep(0.5)


DATA = DataStore()
# !!!!!! widgets
class Gauge(ttk.Frame):
    def __init__(self, parent, width=360, height=200, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.width, self.height = width, height
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        self._needle = None
        self._pct_label = ttk.Label(self, text="— %", font=("Segoe UI", 12, "bold"))
        self._pct_label.grid(row=1, column=0, pady=(8,0))
        self._status = ttk.Label(self, text="Status: —", font=("Segoe UI", 11))
        self._status.grid(row=2, column=0)
        self._draw_static()
        self.set_value(0)

    def _draw_static(self):
        w, h = self.width, self.height
        cx, cy, r = w//2, h-10, min(w, h*2)//2 - 10
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=-60, fill="#16A34A", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=120, extent=-60, fill="#D97706", outline="")
        self.canvas.create_arc(cx-r, cy-r, cx+r, cy+r, start=60, extent=-60, fill="#DC2626", outline="")
        for i in range(0, 11):
            ang = math.radians(180 + i*18)
            x0 = cx + (r-18)*math.cos(ang); y0 = cy + (r-18)*math.sin(ang)
            x1 = cx + (r-2)*math.cos(ang);  y1 = cy + (r-2)*math.sin(ang)
            self.canvas.create_line(x0, y0, x1, y1, width=2, fill="#334155")
        self.canvas.create_text(cx - r + 40, cy - 20, text="OK", font=("Segoe UI", 11, "bold"))
        self.canvas.create_text(cx + r - 40, cy - 20, text="NG", font=("Segoe UI", 11, "bold"))
        self._cx, self._cy, self._r = cx, cy, r

    def set_value(self, pct: float):
        pct = max(0.0, min(100.0, float(pct)))
        ang = math.radians(180 + 180 * pct / 100.0)
        cx, cy, r = self._cx, self._cy, self._r
        x = cx + (r-26) * math.cos(ang)
        y = cy + (r-26) * math.sin(ang)
        if self._needle: self.canvas.delete(self._needle)
        self._needle = self.canvas.create_line(cx, cy, x, y, width=5, fill="#111827", capstyle=tk.ROUND)
        self._pct_label.config(text=f"{pct:0.1f}% confidence")
        status = "NG" if pct >= 50 else "OK"
        self._status.config(text=f"Status: {status}",
                            foreground=("#DC2626" if status=="NG" else "#16A34A"))

class TrendChart(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.bind("<Configure>", lambda e: self.redraw())

    def redraw(self):
        self.canvas.delete("all")
        w = max(200, self.canvas.winfo_width())
        h = max(140, self.canvas.winfo_height())
        pad = 16
        self.canvas.create_rectangle(0,0,w,h, fill="white", outline="")
        self.canvas.create_rectangle(pad,pad,w-pad,h-pad, outline="#CBD5E1")

        y = DATA.recent_window(1200)
        if len(y) < 5:
            self.canvas.create_text(w//2, h//2, text="(Load CSV to see history)", fill="#6B7280")
            return

        ymin, ymax = min(y), max(y)
        if abs(ymax - ymin) < 1e-9:
            ymax = ymin + 1.0

        def X(i):  return pad + (w-2*pad) * (i/(len(y)-1))
        def Y(v):  return h-pad - (h-2*pad) * ((v - ymin)/(ymax - ymin))

        # raw line
        for i in range(1, len(y)):
            self.canvas.create_line(X(i-1), Y(y[i-1]), X(i), Y(y[i]), fill="#93C5FD", width=1)

        # smoothed line
        k = max(5, len(y)//50)
        sm, s = [], 0.0
        for i, v in enumerate(y):
            s += v
            if i >= k: s -= y[i-k]
            sm.append(s / min(i+1, k))
        for i in range(1, len(sm)):
            self.canvas.create_line(X(i-1), Y(sm[i-1]), X(i), Y(sm[i]), fill="#2563EB", width=2)

        # slope badge
        slope = DATA.trend_slope(min(1024, len(y)))
        color = "#DC2626" if slope > 0 else ("#16A34A" if slope < 0 else "#6B7280")
        label = "Uptrend" if slope > 0 else ("Downtrend" if slope < 0 else "Stable")
        self.canvas.create_text(w-pad-70, pad+14, text=label, fill=color, font=("Segoe UI", 10, "bold"))
        ax = w - pad - 30; ay = pad + 28
        dy = -16 if slope > 0 else (16 if slope < 0 else 0)
        self.canvas.create_line(ax-10, ay, ax+10, ay+dy, arrow=tk.LAST, width=3, fill=color)

        if getattr(DATA, "classes", None):
            # store image references to prevent garbage collection
            if not hasattr(self, '_overlay_images'):
                self._overlay_images = []
            self._overlay_images.clear()
            
            y0 = pad + 1
            y1 = h - pad - 1
            n_total = len(DATA.od)
            offset = n_total - len(y)
            
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                i0 = max(0, min(i0, len(y) - 2))     # allow room for at least 1 sample
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                x0 = X(i0)
                x1 = X(i1)
                
                # get color and create semitransparent overlay
                color = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                rgb = self.canvas.winfo_rgb(color)
                # winfo_rgb returns 16-bit values (0-65535), convert to 8-bit (0-255)
                r = rgb[0] >> 8
                g = rgb[1] >> 8
                b = rgb[2] >> 8
                alpha = 80  # transparency (0=transparent, 255=opaque)
                
                # create RGBA image with proper dimensions
                width = int(x1 - x0)
                height = int(y1 - y0)
                
                if width > 0 and height > 0:
                    image = Image.new('RGBA', (width, height), (r, g, b, alpha))
                    photo = ImageTk.PhotoImage(image)
                    self._overlay_images.append(photo)
                    
                    # use x0, y0 with anchor='nw' to position at top-left
                    self.canvas.create_image(x0, y0, image=photo, anchor='nw')
                
                # draw label
                self.canvas.create_text(x0+4, y0+10, text=seg["label"], anchor="w",
                                        fill="#333333", font=("Segoe UI", 8, "bold"))



        # legend
        legend_x = pad + 6
        legend_y = pad + 10
        for idx, (name, col) in enumerate(CLASS_COLORS.items()):
            if name not in VISIBLE_CLASSES: continue
            self.canvas.create_rectangle(legend_x, legend_y + idx*16 - 6,
                                         legend_x+12, legend_y + idx*16 + 6,
                                         fill=col, width=0, stipple="gray25")
            self.canvas.create_text(legend_x+18, legend_y + idx*16, text=name, anchor="w",
                                    fill="#111827", font=("Segoe UI", 8))

# !!! pages
class BasePage(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, padding=16, *args, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(99, weight=1)
    def headline(self, text):
        lbl = ttk.Label(self, text=text, style="Headline.TLabel")
        lbl.grid(row=0, column=0, sticky="w", pady=(0, 12))
        return lbl
    def placeholder(self, parent, text):
        box = ttk.Label(parent, text=text, style="Placeholder.TLabel",
                        anchor="center", padding=24, relief="ridge")
        box.grid(sticky="nsew")
        return box

class DataPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Data")
        controls = ttk.Frame(self); controls.grid(row=1, column=0, sticky="ew", pady=(0,12))
        for i in range(12): controls.columnconfigure(i, weight=1)
        # DataPage.__init__ controls block

        ttk.Button(controls, text="Load XLSX…", command=self.load_xlsx).grid(row=0, column=0, sticky="w", padx=(0,8))
        
        ttk.Label(controls, text="Compare to:").grid(row=0, column=8, sticky="w", padx=(0,4))
        self.sheet_var = tk.StringVar(value="Select sheet...")
        self.sheet_dropdown = ttk.Combobox(controls, textvariable=self.sheet_var, state="disabled", width=25, height=25)
        self.sheet_dropdown.grid(row=0, column=9, sticky="w", padx=(0,8))
        self.sheet_dropdown.bind('<<ComboboxSelected>>', self.on_sheet_select)
        
        ttk.Button(controls, text="Show Correlation", command=self.show_corr).grid(row=0, column=10, sticky="w", padx=(0,8))


        self.info = ttk.Label(self, text="No file loaded.", foreground="#6B7280")
        self.info.grid(row=2, column=0, sticky="w")

        table_area = ttk.Frame(self); table_area.grid(row=3, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(3, weight=1)
        self.placeholder(table_area, "Recent files & preview table will appear here.")

        ttk.Label(controls, text="WebSocket URL:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.ws_url = tk.StringVar(value="ws://localhost:6467")
        ttk.Entry(controls, textvariable=self.ws_url, width=40).grid(row=1, column=1, sticky="w", pady=(6,0), padx=(0,8))

        ttk.Button(controls, text="Connect Live", command=self.connect_live).grid(row=1, column=2, sticky="w", pady=(6,0), padx=(0,8))
        ttk.Button(controls, text="Disconnect",  command=self.disconnect_live).grid(row=1, column=3, sticky="w", pady=(6,0))

        self.decimate_var = tk.BooleanVar(value=False)
        def _on_decimate_toggle(*_):
            # update global setting and clear any partial bucket
            DATA.decimate_enabled = self.decimate_var.get()
            DATA._decim_current_sec = None
            DATA._decim_vals = []
            DATA._decim_speed_ok = False
            App.status(f"Decimation to 1 Hz {'enabled' if DATA.decimate_enabled else 'disabled'}")
        self.decimate_var.trace_add('write', _on_decimate_toggle)
        ttk.Checkbutton(controls, text="Decimate live data to 1 Hz (median)", variable=self.decimate_var).grid(row=1, column=4, sticky="w", pady=(6,0), padx=(8,0))


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
            
            App.status("Data loaded. History & gauge now using real data.")
        except Exception as e:
            messagebox.showerror("Load Data failed", str(e))
            App.status("Data load failed")

    def on_sheet_select(self, event):
        selected_sheet = self.sheet_var.get()
        if selected_sheet and selected_sheet not in ["Select sheet...", "No other sheets"]:
            try:
                DATA.load_secondary_sheet(selected_sheet)
                self.info.config(text=f"{self.info.cget('text')}  •  comparing to '{selected_sheet}' (paired={len(DATA.paired_df)})")
                App.status(f"Secondary sheet loaded: {selected_sheet}")
            except Exception as e:
                messagebox.showerror("Load Secondary failed", str(e))
                App.status("Load secondary failed")
                self.sheet_var.set("Select sheet...")

    def show_corr(self):
        if DATA.paired_df is None or DATA.paired_df.empty:
            messagebox.showinfo("Correlation", "Select a secondary sheet first from the dropdown.")
            return
        CorrelationWindow(self)

    def connect_live(self):
        try:
            DATA.start_live(self.ws_url.get().strip())
            self.info.config(text=f"Live: connected to {self.ws_url.get().strip()}")
            App.status("Live feed connected")
        except Exception as e:
            messagebox.showerror("Live Connect failed", str(e))
            App.status("Live connect failed")

    def disconnect_live(self):
        DATA.stop_live()
        self.info.config(text=f"{self.info.cget('text')}  •  live stopped")
        App.status("Live feed disconnected")

    def _poll_live_queue(self):
        # drain queue and append samples
        got = DATA._consume_live_queue()
        # if a model is selected, we can auto-update classes here
        # for efficiency, do this on a cadence or on a fixed “latest n” buffer, not every single tick.
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
        
        ttk.Label(right_frame, text="Average likelihood of chatter being detected in entire dataset (all windows)\n         by each model over different window sizes\nHigher likelihood = chatter likely present\nLower likelihood = chatter unlikely\nMiddle likelihood near decision boundary = unsure prediction", 
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
            # if hasattr(app_instance, 'pages') and 'Results' in app_instance.pages:
            #     results_page = app_instance.pages['Results']
            #     results_page._tick()
            if hasattr(app_instance, 'pages') and 'Analysis' in app_instance.pages:
                analysis_page = app_instance.pages['Analysis']
                analysis_page.update_confidence_timeline()
            
            App.status(f"Model changed and classifications updated")

    def update_confidence_plot(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return
        
        App.status("Computing confidence curves... this may take a moment")
        
        # group models by type and window size
        model_types = {}  # {model_type: {window_size: model}}
        for model_name, model in self.models.items():
            # parse model name: "Model Type (Window Size XX)"
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
        
        colors = ['#2563EB', '#DC2626', '#16A34A', '#F59E0B']
        markers = ['o', 's', '^', 'd']
        
        for idx, (model_type, (ws, confs)) in enumerate(sorted(results.items())):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            self.ax_conf.plot(ws, confs, marker=marker, linewidth=2, markersize=6,
                            label=model_type, color=color)
        
        self.ax_conf.set_xlabel('Window Size (samples)', fontsize=11)
        self.ax_conf.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax_conf.set_title('Average Chatter Likelihood vs Window Size', 
                              fontsize=12, fontweight='bold')
        self.ax_conf.legend(loc='best', fontsize=9)
        self.ax_conf.grid(True, alpha=0.3)
        self.ax_conf.set_ylim([0, 105])
        
        self.ax_conf.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_conf.text(self.ax_conf.get_xlim()[1], 50, ' Decision boundary', 
                         va='center', fontsize=9, color='gray')
        
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()
        
        App.status(f"Confidence plot updated with {len(results)} model types")

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
        App.status("Confidence plot reset")

class LiveTimeSeries(ttk.Frame):
    """matplotlib live plot with class shading via axvspan."""
    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(6,3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tbar = ttk.Frame(self); tbar.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, tbar)

        self._user_view_active = False
        self._saved_xlim = None
        self._saved_ylim = None

        def _on_mouse(event):
            try:
                mode = getattr(self.toolbar, "mode", "")
            except Exception:
                mode = ""
            if event.inaxes is self.ax and mode in ("zoom rect", "pan/zoom"):
                self._user_view_active = True
                self._saved_xlim = self.ax.get_xlim()
                self._saved_ylim = self.ax.get_ylim()

        self.canvas.mpl_connect("button_release_event", _on_mouse)
        self.canvas.mpl_connect("scroll_event", lambda e: (
            setattr(self, "_user_view_active", True),
            setattr(self, "_saved_xlim", self.ax.get_xlim()),
            setattr(self, "_saved_ylim", self.ax.get_ylim())
        ))

        def _home_wrapper(*args, **kwargs):
            self.reset_view()
            NavigationToolbar2Tk.home(self.toolbar)

        try:
            btn = (getattr(self.toolbar, "_buttons", {}).get("home")
                or getattr(self.toolbar, "_buttons", {}).get("Home"))
            if btn is not None:
                btn.configure(command=_home_wrapper)
        except Exception:
            pass

        self._last_len = -1
        self.toolbar.update()
        self.after(1000, self._tick)
        

    def reset_view(self):
        """return to autoscaling and clear any saved manual limits."""
        self._user_view_active = False
        self._saved_xlim = None
        self._saved_ylim = None
        try:
            self.ax.relim()
            self.ax.autoscale_view()
        except Exception:
            pass
        self.canvas.draw_idle()
        # if you have a toolbar present:
        try:
            self.toolbar.update()
        except Exception:
            pass

    def _draw(self):
        if self._user_view_active:
            xlim = self._saved_xlim or self.ax.get_xlim()
            ylim = self._saved_ylim or self.ax.get_ylim()
        else:
            xlim = ylim = None
        self.ax.clear()
        if not DATA.od:
            self.ax.text(0.5,0.5,"(Load CSV to see live plot)", ha="center", va="center")
            self.ax.axis("off")
            self.canvas.draw(); return

        y = np.asarray(DATA.od[-24000:])  # last ~2400 samples
        x = np.arange(len(y))
        self.ax.plot(x, y, linewidth=1.0, alpha=0.7, label="OD")

        # simple smoothing
        k = max(5, len(y)//50)
        if len(y) >= k:
            sm = pd.Series(y).rolling(k, min_periods=1).mean().values
            self.ax.plot(x, sm, linewidth=2.0, label="smooth")

        # class shading (convert global i0/i1 to local indices)
        if getattr(DATA, "classes", None):
            n_total = len(DATA.od)
            offset = n_total - len(y)
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                # clamp so we always draw something
                i0 = max(0, min(i0, len(y) - 2))
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                c = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                self.ax.axvspan(i0, i1, facecolor=c, alpha=0.25, linewidth=0)
                
                # Add label text on the overlay
                # self.ax.text(i0 + 2, self.ax.get_ylim()[1] * 0.95, seg["label"], 
                #            fontsize=8, color='#333333', weight='bold')

        self.ax.set_title("OD vs Samples — live")
        self.ax.set_xlabel("sample index")
        self.ax.set_ylabel("OD (inches)")
        
        # Create legend with both line plots AND class overlays
        handles, labels = self.ax.get_legend_handles_labels()
        
        # Add class colors to legend
        if getattr(DATA, "classes", None):
            from matplotlib.patches import Patch
            for name, color in CLASS_COLORS.items():
                if name in VISIBLE_CLASSES:
                    handles.append(Patch(facecolor=color, alpha=0.25, label=name))
                    labels.append(name)

        if self._user_view_active and xlim and ylim:
            # Preserve user’s zoom/pan
            try:
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except Exception:
                # If limits invalid due to data shrinking, fall back to autoscale
                self._user_view_active = False
                self._saved_xlim = self._saved_ylim = None
                self.ax.relim(); self.ax.autoscale_view()
        else:
            # No manual interaction: autoscale to new data
            self.ax.relim()
            self.ax.autoscale_view()
        
        self.ax.legend(handles, labels, loc="upper left", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
        self.toolbar.update()

    def _tick(self):
        current_len = len(DATA.od)
        current_classes = len(DATA.classes) if hasattr(DATA, 'classes') else 0
        
        if (current_len != self._last_len or 
            current_classes != getattr(self, '_last_classes_len', -1)):
            self._draw()
            self.toolbar.update()
            self._last_len = current_len
            self._last_classes_len = current_classes
        
        self.after(1000, self._tick)

class ResultsPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Results")

        tools = ttk.Frame(self); tools.grid(row=1, column=0, sticky="ew", pady=(0,12))
        # ttk.Button(tools, text="Predict Latest", command=self.predict_latest).grid(row=0, column=0, sticky="w", padx=(0,8))
        # ttk.Button(tools, text="Export Report", command=lambda: App.busy("Export report (todo)")).grid(row=0, column=1, sticky="w", padx=(0,8))

        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=2)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        left = ttk.Frame(grid); left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0,8))
        ttk.Label(left, text="OD vs Time (Live)", style="Subhead.TLabel").pack(anchor="w")
        self.live = LiveTimeSeries(left)
        self.live.pack(fill="both", expand=True, pady=(6,0))

        right_top = ttk.Frame(grid); right_top.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        ttk.Label(right_top, text="OK / NG Indicator", style="Subhead.TLabel").pack(anchor="w")
        self.gauge = Gauge(right_top, width=360, height=200); self.gauge.pack(fill="both", expand=True, pady=(6,0))

        self.pred_label = ttk.Label(right_top, text="Predicted class: —", style="KPI.TLabel")
        self.pred_label.pack(anchor="w", pady=(6, 0))
        self.pred_conf  = ttk.Label(right_top, text="Confidence: —", style="KPI.TLabel")
        self.pred_conf.pack(anchor="w")

        self.fft_metric = ttk.Label(
            right_top,
            text="FFT: —",
            style="KPI.TLabel"
        )
        self.fft_metric.pack(anchor="w", pady=(2, 0))

        self.after(1000, self._tick)

    def _tick(self):
        pct = 50

        if DATA.od and DATA.classes:
            lbl, class_risk = DATA.current_class()
            if class_risk is not None:
                pct = class_risk * 100.0
                self.pred_label.config(text=f"Predicted class: {lbl}")
                self.pred_conf.config(text=f"Confidence: {pct:0.1f}%")
            else:
                pct = 0
                self.pred_label.config(text="Error")
                self.pred_conf.config(text=f"Error")
        elif DATA.od:
            # data loaded but no model selected yet
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please select a model")
        else:
            # no data - demo mode
            pct = 50 + 15 * math.sin(datetime.now().timestamp()/2.0)
            self.pred_label.config(text="Predicted class: —")
            self.pred_conf.config(text="Please import data and select a model")

        self.gauge.set_value(pct)

        fft_info = self._compute_fft_metric(n=24000)

        if fft_info is None:
            self.fft_metric.config(
                text="FFT: inconclusive (not enough data or flat signal)"
            )
        # elif not fft_info["periodic"]:
        #     # FFT computed but no strong dominant peak
        #     self.fft_metric.config(
        #         text="FFT: inconclusive (no dominant periodic component)"
        #     )
        else:
            # clear harmonic found
            self.fft_metric.config(
                text=(
                    f"FFT: peak power {fft_info['peak_power']:.3g} "
                    f"at {fft_info['peak_freq']:.3g} Hz "
                    f"(prominence {fft_info['prominence']*100:.1f}%)"
                )
            )

        self.after(1000, self._tick)

    def _compute_fft_metric(self, n=512):
        """
        Look at the most recent n OD samples and compute:
          - peak frequency (Hz)
          - peak power (arbitrary units)
          - prominence = peak_power / total_power in non-DC bins
        Returns None if there isn't enough data or signal is degenerate.
        """
        if not DATA.od:
            return None

        y = np.asarray(DATA.recent_window(n), dtype=float)
        if y.size < 16:
            # too short for a meaningful FFT
            return None

        # detrend by removing mean
        y = y - np.mean(y)
        if not np.any(np.isfinite(y)) or np.allclose(y, 0.0, atol=1e-12):
            return None

        # estimate sampling period from timestamps (fallback to 1 Hz)
        fs = 1.0
        try:
            if DATA.ts_dt and len(DATA.ts_dt) >= len(y):
                t = pd.to_datetime(DATA.ts_dt[-len(y):], errors="coerce")
                t = t[~pd.isna(t)]
                if len(t) >= 2:
                    dt_sec = np.median(
                        np.diff(t).astype("timedelta64[ns]").astype(np.float64)
                    ) / 1e9
                    if dt_sec > 0:
                        fs = 1.0 / dt_sec
        except Exception:
            pass

        nfft = len(y)
        freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
        fft_vals = np.fft.rfft(y)
        psd = (np.abs(fft_vals)**2) / nfft

        if psd.size <= 1:
            return None

        # ignore DC component at index 0
        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]

        peak_idx = int(np.argmax(psd_no_dc))
        peak_freq = float(freqs_no_dc[peak_idx])
        peak_power = float(psd_no_dc[peak_idx])
        total_power = float(np.sum(psd_no_dc))

        if not np.isfinite(peak_power) or total_power <= 0 or not np.isfinite(total_power):
            return None

        prominence = float(peak_power / total_power)

        # heuristic: require a reasonably dominant peak
        periodic = prominence > 0.1

        return {
            "fs": fs,
            "peak_freq": peak_freq,
            "peak_power": peak_power,
            "prominence": prominence,
            "periodic": periodic,
        }


class HistoryPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Historical Trend")
        # desc = "Trend view shows last N samples with smoothed curve and class overlays.\n" \
        #        "Use this to see if the process is drifting toward NG before defects happen."
        ttk.Label(self, foreground="#6B7280").grid(row=1, column=0, sticky="w", pady=(0,8))

        grid = ttk.Frame(self); grid.grid(row=2, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        grid.columnconfigure(0, weight=3); grid.columnconfigure(1, weight=2)
        grid.rowconfigure(0, weight=1)

        self.chart = TrendChart(grid); self.chart.grid(row=0, column=0, sticky="nsew", padx=(0,8))

        side = ttk.Frame(grid); side.grid(row=0, column=1, sticky="nsew", padx=(8,0))
        side.columnconfigure(0, weight=1)
        # ttk.Label(side, text="Latest Stats", style="Subhead.TLabel").grid(row=0, column=0, sticky="w", pady=(0,6))
        # self.l_slope = ttk.Label(side, text="Slope: —", style="KPI.TLabel"); self.l_slope.grid(row=1, column=0, sticky="w", pady=4)
        # self.l_p2p   = ttk.Label(side, text="Peak-to-peak: —", style="KPI.TLabel"); self.l_p2p.grid(row=2, column=0, sticky="w", pady=4)
        # self.l_score = ttk.Label(side, text="Risk score: —", style="KPI.TLabel"); self.l_score.grid(row=3, column=0, sticky="w", pady=4)

        self.after(1000, self._tick)

    def _tick(self):
        # self.l_slope.config(text=f"Slope: {slope:0.6f} in/sample")
        # self.l_p2p.config(text=f"Peak-to-peak: {p2p:0.4f} in")
        # self.l_score.config(text=f"Risk score: {score:0.1f}/100")
        self.chart.redraw()
        self.after(1000, self._tick)

class AnalysisPage(BasePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.headline("Modeling & Analysis")

        top = ttk.Frame(self); top.grid(row=1, column=0, sticky="ew", pady=(0,8))
        self.status_lbl = ttk.Label(top, text="", foreground="#6B7280")
        self.status_lbl.grid(row=0, column=0, sticky="w", padx=12)

        area = ttk.Frame(self); area.grid(row=2, column=0, sticky="nsew")
        self.rowconfigure(2, weight=1); self.columnconfigure(0, weight=1)

        # --- Existing figure: Confidence timeline of model output ---
        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._overlay_images = []

        self.ax.text(0.5, 0.5, "Load data and select a model to see predictions",
                     ha="center", va="center", fontsize=12, color='#6B7280')
        self.ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw()

        # --- NEW: FFT figure: latest time window FFT ---
        self.fft_fig = Figure(figsize=(10, 3), dpi=100)
        self.fft_ax = self.fft_fig.add_subplot(111)
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=area)
        self.fft_canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

        self.fft_ax.text(0.5, 0.5, "FFT of latest window will appear here",
                         ha="center", va="center", fontsize=11, color="#6B7280")
        self.fft_ax.set_axis_off()
        self.fft_fig.tight_layout()
        self.fft_canvas.draw()

        # Periodically refresh FFT so live feed is visible
        self.after(1500, self._tick_fft)

    def _tick_fft(self):
        # update FFT plot even if no model is selected
        self.update_fft_plot(n=24000)
        self.after(1500, self._tick_fft)

    def update_confidence_timeline(self):
        if not DATA.od:
            messagebox.showinfo("No Data", "Please load data first")
            return

        if DATA.model is None:
            messagebox.showinfo("No Model", "Please select a model first.")
            return

        App.status("Computing confidence timeline.")

        self._overlay_images.clear()
        self.ax.clear()

        ws = DATA.window_size
        num_windows = len(DATA.od) // ws

        if num_windows < 1:
            self.ax.text(0.5, 0.5, "Not enough data for the selected window size",
                         ha="center", va="center", fontsize=12)
            self.ax.axis("off")
            self.canvas.draw()
            return

        confidences = []
        window_times = []

        if not DATA.ts_dt:
            DATA.ts_dt = pd.to_datetime(DATA.ts, errors='coerce').tolist()
            if not DATA.ts_dt or all(pd.isna(t) for t in DATA.ts_dt):
                DATA.ts_dt = [pd.Timestamp.now() + pd.Timedelta(seconds=i) for i in range(len(DATA.od))]

        for i in range(num_windows):
            start_idx = i * ws
            end_idx = start_idx + ws
            if end_idx > len(DATA.od):
                break

            window = DATA.od[start_idx:end_idx]
            features_dict = DATA.extract_features(window)
            X = pd.DataFrame([features_dict])
            probas = DATA.model.predict_proba(X)
            confidences.append(probas[0, 1] * 100)

            # use timestamp at center of window for x-axis
            mid_idx = start_idx + ws // 2
            if mid_idx < len(DATA.ts_dt):
                window_times.append(DATA.ts_dt[mid_idx])
            else:
                window_times.append(DATA.ts_dt[-1])

        if not confidences:
            self.ax.text(0.5, 0.5, "Not enough data to compute timeline",
                         ha="center", va="center", fontsize=12)
            self.ax.axis("off")
            self.canvas.draw()
            return

        self.ax.plot(window_times, confidences, label="Chatter likelihood", color="#2563EB", linewidth=2)

        # shaded class bands, if any
        for span in getattr(DATA, "classes", []):
            label = span.get("label", "UNCERTAIN")
            color = CLASS_COLORS.get(label, "#BBBBBB")
            start_time = span.get("start")
            end_time = span.get("end")
            if start_time is None or end_time is None:
                continue
            self.ax.axvspan(start_time, end_time,
                            facecolor=pastel(color, 0.25),
                            alpha=0.25,
                            linewidth=0, zorder=0)

        self.ax.set_xlabel('Time', fontsize=11)
        self.ax.set_ylabel('Chatter Likelihood (%)', fontsize=11)
        self.ax.set_title(f'Chatter Confidence in each Window over Time (Window Size: {ws} samples)',
                          fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3, zorder=0)
        self.ax.set_ylim([0, 105])

        # Legend for line + class colors
        handles, labels = self.ax.get_legend_handles_labels()
        from matplotlib.patches import Patch
        for name, color in CLASS_COLORS.items():
            if name in VISIBLE_CLASSES:
                handles.append(Patch(facecolor=pastel(color, 0.25), label=name))
                labels.append(name)
        self.ax.legend(handles, labels, loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

        self.status_lbl.config(text=f"Timeline updated: {len(confidences)} windows analyzed")
        App.status(f"Confidence timeline computed for {len(confidences)} windows")

    def update_fft_plot(self, n=512):
        """
        Plot FFT of the latest n OD samples in the bottom chart.
        Shows 'inconclusive' style messages if there is not enough data or no clear peak.
        """
        self.fft_ax.clear()

        if not DATA.od:
            self.fft_ax.text(0.5, 0.5, "No data loaded",
                             ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        y = np.asarray(DATA.recent_window(n), dtype=float)
        if y.size < 16:
            self.fft_ax.text(0.5, 0.5, "Not enough samples for FFT",
                             ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        y = y - np.mean(y)
        if not np.any(np.isfinite(y)) or np.allclose(y, 0.0, atol=1e-12):
            self.fft_ax.text(0.5, 0.5, "FFT inconclusive (flat signal)",
                             ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        # Estimate sampling rate from OD timestamps (fallback to 1 Hz)
        fs = 1.0
        try:
            if DATA.ts_dt and len(DATA.ts_dt) >= len(y):
                t = pd.to_datetime(DATA.ts_dt[-len(y):], errors="coerce")
                t = t[~pd.isna(t)]
                if len(t) >= 2:
                    dt_sec = np.median(
                        np.diff(t).astype("timedelta64[ns]").astype(np.float64)
                    ) / 1e9
                    if dt_sec > 0:
                        fs = 1.0 / dt_sec
        except Exception:
            pass

        nfft = len(y)
        freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
        fft_vals = np.fft.rfft(y)
        psd = (np.abs(fft_vals)**2) / nfft

        if psd.size <= 1:
            self.fft_ax.text(0.5, 0.5, "FFT inconclusive",
                             ha="center", va="center", fontsize=11)
            self.fft_ax.set_axis_off()
            self.fft_fig.tight_layout()
            self.fft_canvas.draw()
            return

        # Ignore DC bin when plotting and measuring
        freqs_plot = freqs[1:]
        psd_plot = psd[1:]

        self.fft_ax.plot(freqs_plot, psd_plot)
        self.fft_ax.set_xlabel("Frequency (Hz)")
        self.fft_ax.set_ylabel("Power")
        self.fft_ax.set_title("FFT of Latest Window")
        self.fft_ax.grid(True, alpha=0.3)
        self.fft_ax.set_axis_on()

        # Highlight dominant frequency if clearly periodic
        peak_idx = int(np.argmax(psd_plot))
        peak_freq = float(freqs_plot[peak_idx])
        peak_power = float(psd_plot[peak_idx])
        total_power = float(np.sum(psd_plot))
        prominence = peak_power / total_power if total_power > 0 else 0.0

        if total_power > 0 and prominence > 0.30:
            self.fft_ax.axvline(peak_freq, linestyle="--", alpha=0.7)
            self.fft_ax.text(
                peak_freq, peak_power,
                f"{peak_freq:.3g} Hz",
                rotation=90, va="bottom", ha="right", fontsize=9
            )

        self.fft_fig.tight_layout()
        self.fft_canvas.draw()


class CorrelationWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("OD ↔ Secondary Correlation")
        self.minsize(980, 720)

        if DATA.paired_df is None or DATA.paired_df.empty:
            ttk.Label(self, text="Load a secondary file first.", padding=12).pack()
            return

        # === Top stats / controls ===
        top = ttk.Frame(self, padding=12)
        top.pack(side="top", fill="x")

        stats = DATA.corr_stats()
        rtxt = f"r = {stats['pearson_r']:.3f}  |  best lag: {stats['best_lag']} samples (r={stats['r_at_best_lag']:.3f})"
        sign = "POSITIVE" if stats["pearson_r"] >= 0 else "NEGATIVE"
        color = "#16A34A" if stats["pearson_r"] >= 0 else "#DC2626"

        ttk.Label(top, text=f"Paired rows: {stats['n']}", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,20))
        ttk.Label(top, text=rtxt).grid(row=0, column=1, sticky="w", padx=(0,20))
        badge = ttk.Label(top, text=f"Tendency: {sign}", foreground=color, font=("Segoe UI", 10, "bold"))
        badge.grid(row=0, column=2, sticky="w")

        # Controls
        ctrl = ttk.Frame(top)
        ctrl.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8,0))
        ttk.Label(ctrl, text="Max lag (samples):").grid(row=0, column=0, sticky="w")
        self.maxlag_var = tk.IntVar(value=300)
        ttk.Entry(ctrl, textvariable=self.maxlag_var, width=8).grid(row=0, column=1, sticky="w", padx=(6,16))
        ttk.Label(ctrl, text="Rolling window (samples):").grid(row=0, column=2, sticky="w")
        self.win_var = tk.IntVar(value=200)
        ttk.Entry(ctrl, textvariable=self.win_var, width=8).grid(row=0, column=3, sticky="w", padx=(6,16))
        ttk.Button(ctrl, text="Update plots", command=self._refresh_all).grid(row=0, column=4)

        # === Tabs ===
        nb = ttk.Notebook(self)
        nb.pack(side="top", fill="both", expand=True, padx=12, pady=12)

        self.tab_overlay = ttk.Frame(nb); nb.add(self.tab_overlay, text="Time Overlay")
        self.tab_scatter = ttk.Frame(nb); nb.add(self.tab_scatter, text="Scatter & Density")
        self.tab_lag     = ttk.Frame(nb); nb.add(self.tab_lag,     text="Corr vs Lag")
        self.tab_roll    = ttk.Frame(nb); nb.add(self.tab_roll,    text="Rolling Corr")

        # Matplotlib canvases
        self.fig_overlay = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_overlay  = self.fig_overlay.add_subplot(111)
        self.cv_overlay  = FigureCanvasTkAgg(self.fig_overlay, master=self.tab_overlay)
        self.cv_overlay.get_tk_widget().pack(fill="both", expand=True)

        self.fig_scatter = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_scatter  = self.fig_scatter.add_subplot(111)
        self.cv_scatter  = FigureCanvasTkAgg(self.fig_scatter, master=self.tab_scatter)
        self.cv_scatter.get_tk_widget().pack(fill="both", expand=True)

        self.fig_lag = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_lag  = self.fig_lag.add_subplot(111)
        self.cv_lag  = FigureCanvasTkAgg(self.fig_lag, master=self.tab_lag)
        self.cv_lag.get_tk_widget().pack(fill="both", expand=True)

        self.fig_roll = Figure(figsize=(7.5, 4.5), dpi=100)
        self.ax_roll  = self.fig_roll.add_subplot(111)
        self.cv_roll  = FigureCanvasTkAgg(self.fig_roll, master=self.tab_roll)
        self.cv_roll.get_tk_widget().pack(fill="both", expand=True)

        self._refresh_all()

    # ---- drawing helpers ----
    def _refresh_all(self):
        self._draw_overlay()
        self._draw_scatter()
        self._draw_lag()
        self._draw_rolling()

    def _draw_overlay(self):
        df = DATA.paired_df.copy()
        df["od_z"]  = (df["od"]  - df["od"].mean())  / (df["od"].std(ddof=0)  + 1e-12)
        df["sec_z"] = (df["sec"] - df["sec"].mean()) / (df["sec"].std(ddof=0) + 1e-12)

        self.ax_overlay.clear()
        t = pd.to_datetime(df["t"], errors="coerce")
        self.ax_overlay.plot(t, df["od_z"],  linewidth=1.5, label="OD (z-score)")
        self.ax_overlay.plot(t, df["sec_z"], linewidth=1.5, label="Secondary (z-score)")
        self.ax_overlay.axhline(0, linewidth=0.8, color="#999999")
        self.ax_overlay.set_title("Time Overlay (z-scored)")
        self.ax_overlay.set_xlabel("time"); self.ax_overlay.set_ylabel("z-score")
        self.ax_overlay.legend(loc="upper right")
        self.fig_overlay.tight_layout()
        self.cv_overlay.draw()

    def _draw_scatter(self):
        df = DATA.paired_df
        x = df["od"].to_numpy(dtype=float)
        y = df["sec"].to_numpy(dtype=float)

        self.ax_scatter.clear()
        # Use hexbin for “density”; easy to eyeball slope & structure
        hb = self.ax_scatter.hexbin(x, y, gridsize=40, bins="log")
        self.fig_scatter.colorbar(hb, ax=self.ax_scatter, fraction=0.046, pad=0.04, label="log density")

        # Fit line y = a*x + b
        if len(x) >= 2:
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            xx = np.linspace(x.min(), x.max(), 200)
            self.ax_scatter.plot(xx, a*xx + b, linewidth=2, label=f"Fit: y={a:.3f}x+{b:.3f}")

        # Pearson r
        r = float(np.corrcoef(x, y)[0,1]) if len(x) > 1 else np.nan
        self.ax_scatter.set_title(f"Scatter & Density  (r={r:.3f})")
        self.ax_scatter.set_xlabel("OD"); self.ax_scatter.set_ylabel("Secondary")
        self.ax_scatter.legend(loc="best")
        self.fig_scatter.tight_layout()
        self.cv_scatter.draw()

    def _draw_lag(self):
        maxlag = max(5, int(self.maxlag_var.get()))
        lags, r = DATA.lag_corr_curve(max_lag_samples=maxlag)
        self.ax_lag.clear()
        if lags.size:
            self.ax_lag.plot(lags, r, linewidth=2)
            self.ax_lag.axhline(0, linewidth=0.8, color="#999999")
            # annotate best
            k = int(lags[np.nanargmax(r)])
            rmax = float(np.nanmax(r))
            self.ax_lag.axvline(k, linestyle="--", linewidth=1.2)
            self.ax_lag.set_title(f"Correlation vs Lag (best: {k} samples, r={rmax:.3f})")
            self.ax_lag.set_xlabel("lag (samples, + = OD leads)"); self.ax_lag.set_ylabel("Pearson r")
        else:
            self.ax_lag.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_lag.axis("off")
        self.fig_lag.tight_layout()
        self.cv_lag.draw()

    def _draw_rolling(self):
        win = max(20, int(self.win_var.get()))
        t, r = DATA.rolling_corr(win_samples=win, step=max(5, win//10))
        self.ax_roll.clear()
        if r.size:
            self.ax_roll.plot(t, r, linewidth=2)
            self.ax_roll.axhline(0, linewidth=0.8, color="#999999")
            self.ax_roll.set_title(f"Rolling Correlation (window={win} samples)")
            self.ax_roll.set_xlabel("time"); self.ax_roll.set_ylabel("Pearson r")
        else:
            self.ax_roll.text(0.5, 0.5, "Not enough paired data.", ha="center", va="center")
            self.ax_roll.axis("off")
        self.fig_roll.tight_layout()
        self.cv_roll.draw()



# ========================= Main App =========================
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
        self.container = ttk.Frame(root, padding=(12,16,16,16)); self.container.grid(row=0, column=1, sticky="nsew")
        self.container.columnconfigure(0, weight=1); self.container.rowconfigure(0, weight=1)

        self.pages = {
            "Data": DataPage(self.container),
            "Model": ModelPage(self.container),
            "Results": ResultsPage(self.container),
            "History": HistoryPage(self.container),
            "Analysis": AnalysisPage(self.container),
        }
        for p in self.pages.values(): p.grid(row=0, column=0, sticky="nsew")
        self.show("Results")

        self._build_statusbar()
        self.bind("<Control-1>", lambda e: self.show("Data"))
        self.bind("<Control-2>", lambda e: self.show("Results"))
        self.bind("<Control-3>", lambda e: self.show("History"))
        self.bind("<Control-4>", lambda e: self.show("Analysis"))

    def _init_style(self):
        self.style = ttk.Style(self)
        try: self.style.theme_use("clam")
        except tk.TclError: pass
        self.style.configure("Sidebar.TFrame", background="#111827")
        self.style.configure("Sidebar.TButton", foreground="white", background="#1F2937")
        self.style.map("Sidebar.TButton", background=[("active", "#374151")])
        self.style.configure("Headline.TLabel", font=("Segoe UI", 18, "bold"))
        self.style.configure("Subhead.TLabel",  font=("Segoe UI", 12, "bold"))
        self.style.configure("Placeholder.TLabel", foreground="#6B7280", background="white")
        self.style.configure("KPI.TLabel", font=("Segoe UI", 10, "bold"))

    def _init_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_separator(); filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_sidebar(self, parent):
        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=12)
        ttk.Label(bar, text="Wavy Detection", foreground="white", background="#111827",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(0,16))
        for name, accel in [("Data","Ctrl+1"),("Model","Ctrl+2"),("Results","Ctrl+3"),("History","Ctrl+4"),("Analysis","Ctrl+5")]:
            ttk.Button(bar, text=f"{name}    ({accel})", style="Sidebar.TButton",
                       command=lambda n=name: self.show(n)).pack(fill="x", pady=6)
        ttk.Label(bar, text="", background="#111827").pack(expand=True, fill="both")
        ttk.Label(bar, text=f"{APP_VERSION}", foreground="#9CA3AF", background="#111827").pack(anchor="w")
        return bar

    def _build_statusbar(self):
        self._status_var = tk.StringVar(value="Ready")
        bar = ttk.Frame(self); bar.pack(side="bottom", fill="x")
        ttk.Label(bar, textvariable=self._status_var, padding=8).pack(side="left")
        ttk.Label(bar, text=datetime.now().strftime("%Y-%m-%d"), padding=8).pack(side="right")

    def show(self, page_name: str):
        self.pages[page_name].tkraise(); self.status(f"Showing {page_name}")

    @classmethod
    def status(cls, msg: str):
        if cls._status_var is not None: cls._status_var.set(msg)

    @classmethod
    def busy(cls, msg: str):
        cls.status(msg)

if __name__ == "__main__":
    App().mainloop()
