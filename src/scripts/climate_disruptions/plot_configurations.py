"""
plot_configurations.py
"""
import matplotlib.pyplot as plt

# ── Font sizes ────────────────────────────────────────────────────────────────
FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16
FS_LEGEND = 12
FS_PANEL = 17
FS_SUPTITLE = 14

# ── Scenario colours ──────────────────────────────────────────────────────────
SCENARIO_COLOURS = ["#ADB993", "#EDC7CF", "#6F8AB7"]


# ── Shared style ──────────────────────────────────────────────────────────────
def apply_style():
    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_SUPTITLE,

        # Font weight
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",

        # Spines — remove top and right everywhere
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Grid — consistent subtle grid on x or y where used
        "axes.grid": False,  # off by default; enable per-plot where needed
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Layout and output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": False,  # use tight_layout() explicitly
    })
