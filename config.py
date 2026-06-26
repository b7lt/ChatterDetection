APP_TITLE   = "Chatter Detection Dashboard"
APP_VERSION = "v1.1"

# Long-running dashboards need explicit retention limits. Raw OD samples stay at
# full resolution for live plots and inference; history is already downsampled
# to 1 Hz, so it can keep a longer window at modest memory cost.
LIVE_QUEUE_MAX = 10_000
LIVE_QUEUE_DRAIN_LIMIT = 5_000
LIVE_SAMPLE_LIMIT = 500_000
LIVE_SAMPLE_TRIM_TO = 400_000
HISTORY_SAMPLE_LIMIT = 7 * 24 * 60 * 60

OD_TAG = "NDC_System_OD_Value"
FOOTAGE_TAG = "FtCounters_AirRampFootage_Total"
PRESSURE_TAG = "AirRampPressure_Val"
SPEED_TAG_CANDIDATES = [
    "YS_Pullout1_Act_Speed_fpm",
]
RUNNING_SPEED_MIN = 1.0
RUNNING_FOOTAGE_MIN = 0.01
RUNNING_PRESSURE_MIN = 0.001
SEGMENT_FOOTAGE_RESET_DROP = 5.0
SEGMENT_IDLE_AFTER_SECONDS = 2.0

SECONDARY_COL_GUESSES = {
    "time": ["ts", "time", "timestamp", "date_time", "datetime", "t_stamp"],
    "val": [
        "ovality", "ovality_value",
        "ndc_system_ovality_value", "ndc_system_ovality_value__tag_value",
        "tag_value", "value"
    ],
}

CLASS_COLORS = {
    "No Chatter":    "#16A34A",
    "Mild Chatter":  "#D97706",
    "Heavy Chatter": "#DC2626",
}

VISIBLE_CLASSES = set(CLASS_COLORS.keys())


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


def pick(colnames, candidates):
    low = [c.lower() for c in colnames]
    for alias in candidates:
        if alias.lower() in low:
            return colnames[low.index(alias.lower())]
    return None
