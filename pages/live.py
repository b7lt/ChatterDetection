from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from config import CLASS_COLORS, VISIBLE_CLASSES
from data_store import DATA


class LiveTimeSeries(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 3), dpi=100)
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
        """Return to autoscaling and clear any saved manual limits."""
        self._user_view_active = False
        self._saved_xlim = None
        self._saved_ylim = None
        try:
            self.ax.relim()
            self.ax.autoscale_view()
        except Exception:
            pass
        self.canvas.draw_idle()
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
            self.ax.text(0.5, 0.5, "(Load Data to see live plot)", ha="center", va="center")
            self.ax.axis("off")
            self.canvas.draw(); return

        y = np.asarray(DATA.od[-72000:])
        x = np.arange(len(y))
        self.ax.plot(x, y, linewidth=1.0, alpha=0.7, label="OD")

        # simple smoothing — numpy cumsum is much cheaper than pd.rolling
        k = max(5, len(y) // 50)
        if len(y) >= k:
            cs = np.cumsum(np.insert(y, 0, 0.0))
            sm = (cs[k:] - cs[:-k]) / k
            # pad the leading edge so lengths match
            pad = np.full(k - 1, sm[0])
            sm = np.concatenate([pad, sm])
            self.ax.plot(x, sm, linewidth=2.0, label="smooth")

        # class shading (convert absolute i0/i1 to local indices in y)
        if getattr(DATA, "classes", None):
            n_total = len(DATA.od) + getattr(DATA, '_trim_offset', 0)
            offset = n_total - len(y)
            for seg in DATA.classes:
                if seg["label"] not in VISIBLE_CLASSES:
                    continue
                i0 = seg["i0"] - offset
                i1 = seg["i1"] - offset
                if i1 <= 0 or i0 >= len(y):
                    continue
                i0 = max(0, min(i0, len(y) - 2))
                i1 = max(i0 + 1, min(i1, len(y) - 1))

                c = CLASS_COLORS.get(seg["label"], "#BBBBBB")
                self.ax.axvspan(i0, i1, facecolor=c, alpha=0.25, linewidth=0)

        self.ax.set_title("OD of Incoming Samples")
        self.ax.set_xlabel("sample index")
        self.ax.set_ylabel("OD (inches)")

        # Build legend with both line plots AND class overlays
        handles, labels = self.ax.get_legend_handles_labels()

        if getattr(DATA, "classes", None):
            from matplotlib.patches import Patch
            for name, color in CLASS_COLORS.items():
                if name in VISIBLE_CLASSES:
                    handles.append(Patch(facecolor=color, alpha=0.25, label=name))
                    labels.append(name)

        if self._user_view_active and xlim and ylim:
            try:
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            except Exception:
                self._user_view_active = False
                self._saved_xlim = self._saved_ylim = None
                self.ax.relim(); self.ax.autoscale_view()
        else:
            self.ax.relim()
            self.ax.autoscale_view()

        self.ax.legend(handles, labels, loc="upper left", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _tick(self):
        current_len = len(DATA.od)
        current_classes = len(DATA.classes) if hasattr(DATA, 'classes') else 0

        if (current_len != self._last_len or
                current_classes != getattr(self, '_last_classes_len', -1)):
            self._draw()
            self._last_len = current_len
            self._last_classes_len = current_classes

        self.after(1000, self._tick)
