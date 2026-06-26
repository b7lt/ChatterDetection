# Chatter Detection Dashboard Technical Guide

This dashboard is a Tkinter application for loading OD process data, training or selecting a chatter model, viewing live predictions, and reviewing historical process context.

## Running the Dashboard

Install dependencies in a virtual environment, then start the app from the repository root:

```bash
pip install -r requirements.txt
python dashboard.py
```

Use the Model page to import saved PyTorch models from one or more files, or from a selected folder. Model files should be `.pt` or `.pth` files that deserialize to either `ChatterCNN` or `HybridChatterNet` from `models.py`.

Model import and training require PyTorch in the same interpreter that launches
the dashboard. If VSCode can load models but a terminal run cannot, check the
terminal interpreter with `python -c "import sys; print(sys.executable)"` and
install dependencies into that environment.

## Main Workflow

1. Open `Data`, load an Excel file, or connect to a WebSocket feed.
2. Open `Model`, import one or more models, then select the active model and window size. The default production window is `2400` samples.
3. Open `Live` to monitor the current OD trace, chatter gauge, KPIs, and context attribution.
4. Open `History` to review the 1 Hz downsampled OD, context signals, and classified windows.
5. Use `Training` to import labeled XLSX data or a training WebSocket stream, label windows, train a model, and save it back to disk.

## Excel Input Standard

The primary OD sheet must be named:

```text
NDC_System_OD_Value
```

Segment/running detection uses configurable signal names from `config.py`.
The defaults are:

```text
SPEED_TAG_CANDIDATES = ["YS_Pullout1_Act_Speed_fpm"]
FOOTAGE_TAG = "FtCounters_AirRampFootage_Total"
PRESSURE_TAG = "AirRampPressure_Val"
```

Each sheet is expected to include:

```text
t_stamp
Tag_value
```

Rows are treated as running when at least one configured segment signal indicates
movement or an active section. Speed above `RUNNING_SPEED_MIN` is one possible
signal, but footage and air-ramp pressure can also drive segment detection.
Optional context sheets should be named exactly as the model context tags:

```text
AirRampPressure_Val
FtCounters_AirRampFootage_Total
OilHeater_DeliveryTemp_F
OilHeater_ReturnTemp_F
PTs_PT_300_Val
PTs_PT_400_Val
```

Missing context sheets are filled with `0.0`.

Large `.xlsx` files load fastest when `python-calamine` is installed. The
dashboard uses that engine automatically and falls back to `openpyxl` if it is
not available. For repeated large datasets, exporting the same sheet layout to a
columnar format such as Parquet would be materially faster than Excel, but the
current UI is still XLSX-oriented.

## WebSocket Input Standard

The live dashboard accepts either one sample per message or a batch under `samples`.

Single sample:

```json
{
  "t_stamp": "2026-06-26T12:34:56.123Z",
  "NDC_System_OD_Value": 0.06912,
  "YS_Pullout1_Act_Speed_fpm": 42.5,
  "AirRampPressure_Val": 1.85,
  "FtCounters_AirRampFootage_Total": 1032.0,
  "OilHeater_DeliveryTemp_F": 252.4,
  "OilHeater_ReturnTemp_F": 250.9,
  "PTs_PT_300_Val": 25.1,
  "PTs_PT_400_Val": 28.2
}
```

Batched samples:

```json
{
  "samples": [
    {
      "t_stamp": 1782491696.123,
      "NDC_System_OD_Value": 0.06912,
      "YS_Pullout1_Act_Speed_fpm": 42.5
    },
    {
      "t_stamp": 1782491696.124,
      "NDC_System_OD_Value": 0.06909,
      "YS_Pullout1_Act_Speed_fpm": 42.4
    }
  ]
}
```

Field rules:

- `NDC_System_OD_Value` is required. Samples without a finite OD value are rejected.
- Segment/running signals are configurable in `config.py`. The default signals are speed, section footage, and air-ramp pressure. Live OD samples are appended only while the segment tracker is in `running` state.
- `t_stamp` is optional. It may be ISO-8601 text, epoch seconds, or epoch milliseconds. If omitted, receive time is used; batched messages are spread backward at `2400 Hz`.
- Numeric values may be JSON numbers or numeric strings.
- Missing context fields default to `0.0`.

The training WebSocket accepts the same field names and finite numeric values. It stores all numeric fields except `t_stamp` for labeling and training.

## Retention and Performance

Retention limits live in `config.py`:

```python
LIVE_QUEUE_MAX = 10_000
LIVE_QUEUE_DRAIN_LIMIT = 5_000
LIVE_SAMPLE_LIMIT = 500_000
LIVE_SAMPLE_TRIM_TO = 400_000
HISTORY_SAMPLE_LIMIT = 7 * 24 * 60 * 60
```

Raw live OD and context buffers are trimmed from `500,000` samples back to `400,000` samples. Classification windows that fall fully before the retained raw data are trimmed at the same time.

Historical charts use 1 Hz median samples and keep seven days by default. That gives the dashboard a long operational view without retaining every high-rate OD sample indefinitely.

The UI drains at most `5,000` queued live samples per poll. If the receive thread outruns the UI, dropped and rejected sample counts are surfaced in the status bar.

## Model Notes

`ChatterCNN` consumes only normalized OD windows. `HybridChatterNet` consumes normalized OD windows plus the per-window mean of the context tags listed above. Hybrid models store `ctx_mean` and `ctx_std` on the model object so inference can apply the same normalization used during training.

For consistent predictions, use the same window size at inference that was used during training.

Hybrid model context dimensions must also match the dashboard's current
`CONTEXT_TAGS` list. If a model was trained with more or fewer context variables
than the dashboard provides, the app will reject inference and report the
expected versus provided context width.
