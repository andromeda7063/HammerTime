"""
Broadcast Theme & Palette for HammerTime
Standard: F1 Broadcast Design Guidelines
Colors: Carbon Dark #15151E, F1 Red #E10600, White #FFFFFF, DRS Cyan #00D2BE
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Official broadcast colors
CARBON_DARK = "#15151E"
CARD_BG = "#1F1F2B"
F1_RED = "#E10600"
PURE_WHITE = "#FFFFFF"
TEXT_MUTED = "#949498"
DRS_CYAN = "#00D2BE"
TRAIN_YELLOW = "#FF8700"
CLEAN_GREEN = "#10B981"

# F1 2023 Team Color Map
TEAM_COLORS = {
    "VER": "#3671C6", "PER": "#3671C6",  # Red Bull
    "HAM": "#6CD3BF", "RUS": "#6CD3BF",  # Mercedes
    "LEC": "#F91536", "SAI": "#F91536",  # Ferrari
    "NOR": "#F58020", "PIA": "#F58020",  # McLaren
    "ALO": "#358C75", "STR": "#358C75",  # Aston Martin
    "GAS": "#2293D1", "OCO": "#2293D1",  # Alpine
    "ALB": "#37BEDD", "SAR": "#37BEDD",  # Williams
    "TSU": "#5E8FAA", "RIC": "#5E8FAA",  # AlphaTauri / RB
    "BOT": "#C92D4B", "ZHO": "#C92D4B",  # Alfa Romeo / Kick
    "MAG": "#B6BABD", "HUL": "#B6BABD",  # Haas
}


def apply_broadcast_theme():
    """Configures matplotlib with dark carbon broadcast styling."""
    plt.style.use("dark_background")
    mpl.rcParams["figure.facecolor"] = CARBON_DARK
    mpl.rcParams["axes.facecolor"] = CARD_BG
    mpl.rcParams["text.color"] = PURE_WHITE
    mpl.rcParams["axes.labelcolor"] = PURE_WHITE
    mpl.rcParams["xtick.color"] = TEXT_MUTED
    mpl.rcParams["ytick.color"] = TEXT_MUTED
    mpl.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "Helvetica", "sans-serif"]
    mpl.rcParams["axes.edgecolor"] = "#2D2D3D"
    mpl.rcParams["grid.color"] = "#2D2D3D"
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.alpha"] = 0.5
