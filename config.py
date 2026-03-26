APP_TITLE   = "Wavy Detection Dashboard"
APP_VERSION = "v1.0"

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
