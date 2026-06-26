# Chatter Detection Dashboard

Tkinter dashboard for loading OD process data, training or selecting chatter
models, monitoring live WebSocket data, and reviewing historical predictions.

## Quick Start

```bash
pip install -r requirements.txt
python dashboard.py
```

Use the Model page to import one or more `.pt` or `.pth` model files, or load
all models from a selected folder.

## Documentation

See [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) for dashboard usage,
Excel input expectations, the WebSocket payload standard, retention limits, and
model notes.
