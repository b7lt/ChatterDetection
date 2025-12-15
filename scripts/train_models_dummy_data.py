import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight
import xgboost as xgb
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import multiprocessing as mp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic data generation (OD in inches, speed in fpm)
# ---------------------------------------------------------------------------

def generate_synthetic_od_series(
    fs,
    duration_s,
    mean_od_in,
    speed_fpm,
    drift_per_ft_in,
    chatter_wavelengths_in,
    chatter_amps_in,
    noise_std_in,
):
    """
    Generate a synthetic OD series (inches) for a constant-speed run.
    Returns a DataFrame with columns ["t_stamp", "Tag_value", "speed_value"].
    """
    chatter_wavelengths_in = chatter_wavelengths_in or []
    chatter_amps_in = chatter_amps_in or [0.0] * len(chatter_wavelengths_in)
    assert len(chatter_wavelengths_in) == len(chatter_amps_in)

    N = int(fs * duration_s)
    t = np.arange(N) / fs

    speed_value = np.full(N, speed_fpm, dtype=float)
    speed_in_s = speed_fpm * (12.0 / 60.0)  # 1 fpm = 0.2 in/s

    # Length along tube [in] and [ft]
    length_in = np.cumsum(np.full(N, speed_in_s)) / fs
    length_ft = length_in / 12.0

    # Base OD: mean + slow drift
    od_in = mean_od_in + drift_per_ft_in * length_ft

    # Chatter: spatial → temporal via f = v / λ
    od_chatter = np.zeros_like(od_in)
    for lam_in, amp_in in zip(chatter_wavelengths_in, chatter_amps_in):
        if lam_in <= 0:
            continue
        freq_hz = speed_in_s / lam_in
        phase = 2.0 * np.pi * freq_hz * t
        od_chatter += amp_in * np.sin(phase)
    od_in += od_chatter

    # Noise
    od_in += np.random.normal(0.0, noise_std_in, size=N)

    t0 = pd.Timestamp.now()
    t_stamps = t0 + pd.to_timedelta(t, unit="s")

    df = pd.DataFrame(
        {
            "t_stamp": t_stamps,
            "Tag_value": od_in,
            "speed_value": speed_value,
        }
    )
    return df


def make_synthetic_good_bad_multi(
    fs=2400.0,
    duration_s=180.0,
    means=(0.50, 0.75, 1.00),
    n_runs_per_class=3,
):
    good_runs = []
    bad_runs = []

    # Build list of tasks: (mean_od_in, is_good)
    tasks = []
    for mean_od_in in means:
        for _ in range(n_runs_per_class):
            tasks.append((mean_od_in, True, fs, duration_s))   # good
            tasks.append((mean_od_in, False, fs, duration_s))  # bad

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(_generate_single_run, tasks))

    for df, is_good in results:
        if is_good:
            good_runs.append(df)
        else:
            bad_runs.append(df)

    df_good = pd.concat(good_runs, ignore_index=True)
    df_bad = pd.concat(bad_runs, ignore_index=True)

    # Only keep segments where the line is moving
    speed_threshold = 1.0
    df_good_merged = df_good[df_good["speed_value"] > speed_threshold].copy()
    df_bad_merged = df_bad[df_bad["speed_value"] > speed_threshold].copy()

    return df_good_merged, df_bad_merged

def _generate_single_run(args):
    """Worker function for parallel data generation."""
    mean_od_in, is_good, fs, duration_s = args
    base_amp = 0.002 * (mean_od_in / 0.50)

    if is_good:
        df = generate_synthetic_od_series(
            fs=fs, duration_s=duration_s, mean_od_in=mean_od_in,
            speed_fpm=150.0, drift_per_ft_in=2.0e-05,
            chatter_wavelengths_in=[1.0, 2.0],
            chatter_amps_in=[0.25 * base_amp, 0.15 * base_amp],
            noise_std_in=0.00002,
        )
    else:
        df = generate_synthetic_od_series(
            fs=fs, duration_s=duration_s, mean_od_in=mean_od_in,
            speed_fpm=150.0, drift_per_ft_in=5.0e-05,
            chatter_wavelengths_in=[0.5, 1.0, 2.0],
            chatter_amps_in=[1.5 * base_amp, 1.0 * base_amp, 0.7 * base_amp],
            noise_std_in=0.00005,
        )
    return df, is_good

# ---------------------------------------------------------------------------
# Feature extraction (time domain + FFT, mean-aware)
# ---------------------------------------------------------------------------

def extract_features(window_data, fs):
    """
    Extract hand-crafted features from a 1D window of OD values.

    Includes both:
      - absolute features (mean, std, range ...)
      - relative features (normalized by mean)
      - FFT features (peak freq, prominence, valid flag)

    If FFT is not meaningful, fft_valid=0 and FFT numeric features are 0.
    """
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

    # ---- FFT features ----
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


def create_windows(df, label, window_size, fs):
    """
    Slide non-overlapping windows over the OD values in df and extract features.
    """
    features_list = []
    labels_list = []

    od_values = df["Tag_value"].values
    num_windows = len(od_values) // window_size

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = od_values[start_idx:end_idx]
        if len(window) < window_size:
            continue

        feats = extract_features(window, fs=fs)
        features_list.append(feats)
        labels_list.append(label)

    return features_list, labels_list


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model_on_window_size(df_good, df_bad, window_size, fs):
    good_features, good_labels = create_windows(df_good, label=0, window_size=window_size, fs=fs)
    bad_features, bad_labels = create_windows(df_bad, label=1, window_size=window_size, fs=fs)

    print(f"WS {window_size}: {len(good_features)} good windows and {len(bad_features)} bad windows")

    all_features = good_features + bad_features
    all_labels = good_labels + bad_labels
    if not all_features:
        print(f"Skipping ws={window_size}: not enough windows.")
        return {}

    X = pd.DataFrame(all_features)
    y = np.array(all_labels)

    sample_weights = class_weight.compute_sample_weight(
        class_weight="balanced",
        y=y,
    )

    models = {
        "logit": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "svm": SVC(kernel="rbf", random_state=42, probability=True, class_weight="balanced"),
        "xgboost": xgb.XGBClassifier(),
    }

    models_out = {}
    for model_name, model in models.items():
        if model_name == "xgboost":
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
        models_out[model_name] = model

    return models_out


def main():
    fs = 2400.0

    # many means, many runs per class
    df_good_merged, df_bad_merged = make_synthetic_good_bad_multi(
        fs=fs,
        duration_s=36000.0,
        means=(0.50, 1.00),
        n_runs_per_class=4,
    )

    window_sizes = [1200, 2400, 4800]  # in samples

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), "models_dummy_multi")
    os.makedirs(models_dir, exist_ok=True)

    # Prepare training tasks: (window_size, model_name)
    model_names = ["logit", "rf", "svm", "xgboost"]
    tasks = [(ws, name) for ws in window_sizes for name in model_names]

    # Pre-compute features for each window size (avoid redundant work)
    window_data = {}
    for ws in window_sizes:
        good_features, good_labels = create_windows(df_good_merged, label=0, window_size=ws, fs=fs)
        bad_features, bad_labels = create_windows(df_bad_merged, label=1, window_size=ws, fs=fs)
        
        print(f"WS {ws}: {len(good_features)} good windows and {len(bad_features)} bad windows")
        
        all_features = good_features + bad_features
        all_labels = good_labels + bad_labels
        
        if all_features:
            X = pd.DataFrame(all_features)
            y = np.array(all_labels)
            sample_weights = class_weight.compute_sample_weight(class_weight="balanced", y=y)
            window_data[ws] = (X, y, sample_weights)

    # Worker function for training a single model
    def train_single_model(task):
        ws, model_name = task
        if ws not in window_data:
            return None
        
        X, y, sample_weights = window_data[ws]
        
        if model_name == "logit":
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        elif model_name == "svm":
            model = SVC(kernel="rbf", random_state=42, probability=True, class_weight="balanced")
        elif model_name == "xgboost":
            model = xgb.XGBClassifier()
        
        if model_name == "xgboost":
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
        
        return (ws, model_name, model)

    # train models in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(train_single_model, tasks))

    # save models
    for result in results:
        if result is None:
            continue
        ws, model_name, model = result
        filename = f"dummy_{model_name}_ws{ws}.pkl"
        model_path = os.path.join(models_dir, filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {model_name} (ws={ws}) to {model_path}")


if __name__ == "__main__":
    main()
