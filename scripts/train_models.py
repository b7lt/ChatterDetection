import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

speed_threshold = 1

good_datasets = [
    "../../data/tagHistoryData_09-01-00_09-09-09.xlsx",
    # add more good data files here:
    # "../../data/another_good_dataset.xlsx",
]

bad_datasets = [
    "../../data/tagHistoryData_09-24-00_09-26-17 (rejected sample).xlsx",
    # add more bad data files here:
    # "../../data/another_bad_dataset.xlsx",
]

def load_and_merge_dataset(filepath):
    """load OD and speed data from a single Excel file and merge them."""
    df_od = pd.read_excel(filepath, sheet_name="NDC_System_OD_Value")
    df_od['t_stamp'] = pd.to_datetime(df_od['t_stamp'])
    
    df_speed = pd.read_excel(filepath, sheet_name="YS_Pullout1_Act_Speed_fpm")
    df_speed['t_stamp'] = pd.to_datetime(df_speed['t_stamp'])
    df_speed = df_speed.rename(columns={'Tag_value': 'speed_value'})
    
    df_merged = pd.merge(
        df_od, 
        df_speed[['t_stamp', 'speed_value']], 
        on="t_stamp", 
        how="inner"
    )
    df_merged = df_merged[df_merged['speed_value'] > speed_threshold].copy()
    
    return df_merged

def load_multiple_datasets(dataset_list):
    """load and concatenate multiple datasets."""
    all_data = []
    for filepath in dataset_list:
        print(f"Loading: {filepath}")
        df = load_and_merge_dataset(filepath)
        all_data.append(df)
        print(f"  Loaded {len(df)} rows")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows after combining: {len(combined_df)}\n")
    return combined_df

# load all good and bad datasets
print("Loading good datasets...")
df_good_merged = load_multiple_datasets(good_datasets)

print("Loading bad datasets...")
df_bad_merged = load_multiple_datasets(bad_datasets)

def extract_features(window_data):
    mean_val = np.mean(window_data)
    std_val = np.std(window_data)
    # div by zero
    if mean_val == 0:
        mean_val = 1e-10
    
    # features are normalized to the mean here to take into account spans of multiple different tubing target sizes
    features = {
        'coef_variation': std_val / mean_val,  # coefficient of variation ak relative std
        'relative_range': np.ptp(window_data) / mean_val,  # range relative to mean
        'normalized_variance': np.var(window_data) / (mean_val ** 2),
        'peak_to_peak_ratio': np.ptp(window_data) / np.abs(mean_val),
        'relative_max_deviation': (np.max(window_data) - mean_val) / mean_val,
        'relative_min_deviation': (mean_val - np.min(window_data)) / mean_val,
        # differences between consecutive points, indicates smoothness
        'mean_abs_diff': np.mean(np.abs(np.diff(window_data))) / mean_val,
        'max_abs_diff': np.max(np.abs(np.diff(window_data))) / mean_val,
    }
    
    return features


def create_windows(df, label, window_size):
    features_list = []
    labels_list = []
    
    od_values = df['Tag_value'].values
    num_windows = len(od_values) // window_size
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = od_values[start_idx:end_idx]
        
        features = extract_features(window)
        features_list.append(features)
        labels_list.append(label)
    
    return features_list, labels_list

def train_model_on_window_size(df_good, df_bad, window_size):
    good_features, good_labels = create_windows(df_good, label=0, window_size=window_size)
    bad_features, bad_labels = create_windows(df_bad, label=1, window_size=window_size)
    
    print(f"WS {window_size}: {len(good_features)} good windows and {len(bad_features)} bad windows")
    
    all_features = good_features + bad_features
    all_labels = good_labels + bad_labels
    X = pd.DataFrame(all_features)
    y = np.array(all_labels)

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y
    )
    
    models = {
        'logit': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'svm': SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced'),
        'xgboost': xgb.XGBClassifier()
    }
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    models_out = {}
    for model_name, model in models.items():
        if model_name == 'xgboost':
            model.fit(X, y, sample_weight=classes_weights)
        else:
            model.fit(X, y)
        models_out[model_name] = model
    
    return models_out
    
window_sizes = [5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 120]
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(os.path.dirname(script_dir), "models")
os.makedirs(models_dir, exist_ok=True)
for ws in window_sizes:
    models_out = train_model_on_window_size(df_good_merged, df_bad_merged, ws)
    for model_name, model in models_out.items():
        filename = f"model_{model_name}_ws{ws}.pkl"
        model_path = os.path.join(models_dir, filename)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)