"""
Broadcast Insight Cards Generator
Generates 8-12 broadcast-grade graphic and ASCII cards for television overlays.
Palette: Carbon Dark #15151E, F1 Red #E10600, Pure White #FFFFFF.
Standard: ASD-STE100 Plain English
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Card Definitions for Race Scenarios
SAMPLE_BROADCAST_CARDS = [
    {
        "id": "CARD_01",
        "category": "DIRTY AIR",
        "title": "TRAFFIC ALERT",
        "driver_pos": "P3",
        "driver": "VER",
        "ahead_pos": "P2",
        "ahead": "HAM",
        "gap": 0.7,
        "projected_loss_lap": 0.42,
        "train_info": "DRS Train: 4 cars · Sector 2 · Lap 31",
        "badge_color": "#E10600"
    },
    {
        "id": "CARD_02",
        "category": "DRS TRAIN ACCUMULATION",
        "title": "TRAFFIC BOTTLENECK",
        "driver_pos": "P6",
        "driver": "LEC",
        "ahead_pos": "P5",
        "ahead": "SAI",
        "gap": 0.55,
        "projected_loss_lap": 0.68,
        "train_info": "Chain: SAI -> LEC -> NOR -> PIA · Sector 1 · Lap 24",
        "badge_color": "#E10600"
    },
    {
        "id": "CARD_03",
        "category": "CLEAN AIR WINDOW",
        "title": "OVERCUT OPPORTUNITY",
        "driver_pos": "P1",
        "driver": "NOR",
        "ahead_pos": "PIT",
        "ahead": "TRACK CLEAR",
        "gap": 8.4,
        "projected_loss_lap": 0.00,
        "train_info": "Pit Window Clear · Delta loss: 0.00 s · Lap 38",
        "badge_color": "#10B981"
    },
    {
        "id": "CARD_04",
        "category": "BLUE FLAG INTERFERENCE",
        "title": "BACKMARKER CONGESTION",
        "driver_pos": "P2",
        "driver": "PER",
        "ahead_pos": "P18",
        "ahead": "SAR",
        "gap": 1.1,
        "projected_loss_lap": 0.35,
        "train_info": "Lapping backmarker · Mini-Sector 14 · Lap 44",
        "badge_color": "#FF9900"
    },
    {
        "id": "CARD_05",
        "category": "UNDER-CUT CRITICALITY",
        "title": "STRATEGY ALERT",
        "driver_pos": "P4",
        "driver": "RUS",
        "ahead_pos": "P3",
        "ahead": "VER",
        "gap": 1.45,
        "projected_loss_lap": 0.28,
        "train_info": "Turbulent wake increasing tyre degradation · Lap 20",
        "badge_color": "#FF9900"
    },
    {
        "id": "CARD_06",
        "category": "DIRTY AIR",
        "title": "AERODYNAMIC LOSS",
        "driver_pos": "P8",
        "driver": "ALO",
        "ahead_pos": "P7",
        "ahead": "GAS",
        "gap": 0.65,
        "projected_loss_lap": 0.52,
        "train_info": "Midfield pack compression · Curva Grande · Lap 17",
        "badge_color": "#E10600"
    },
    {
        "id": "CARD_07",
        "category": "DRS TRAIN FORMATION",
        "title": "LOCOMOTIVE DETECTED",
        "driver_pos": "P9",
        "driver": "TSU",
        "ahead_pos": "P8",
        "ahead": "ALO",
        "gap": 0.48,
        "projected_loss_lap": 0.74,
        "train_info": "DRS chain active · Cumulative penalty +0.74 s/lap · Lap 29",
        "badge_color": "#E10600"
    },
    {
        "id": "CARD_08",
        "category": "PIT EXIT TRAFFIC",
        "title": "TRAFFIC RELEASE REJOIN",
        "driver_pos": "P5",
        "driver": "HAM",
        "ahead_pos": "P12",
        "ahead": "OCO",
        "gap": 1.85,
        "projected_loss_lap": 0.19,
        "train_info": "Rejoined into midfield wake · Estimated delay 0.38 s · Lap 34",
        "badge_color": "#FF9900"
    },
    {
        "id": "CARD_09",
        "category": "CLEAN AIR GAIN",
        "title": "UNDISTURBED PACE",
        "driver_pos": "P3",
        "driver": "PIA",
        "ahead_pos": "CLEAR",
        "ahead": "GAP > 5.2 s",
        "gap": 5.2,
        "projected_loss_lap": 0.00,
        "train_info": "Free air running · Clean baseline pace locked · Lap 41",
        "badge_color": "#10B981"
    },
    {
        "id": "CARD_10",
        "category": "HIGH-CONGESTION RECOVERY",
        "title": "DRS BREAKAWAY",
        "driver_pos": "P7",
        "driver": "ALB",
        "ahead_pos": "P6",
        "ahead": "BOT",
        "gap": 0.88,
        "projected_loss_lap": 0.38,
        "train_info": "Straight-line defense · Wake loss in corners · Lap 36",
        "badge_color": "#FF9900"
    }
]


def render_ascii_card(c: dict) -> str:
    """Renders broadcast card in exact ASCII box format."""
    cat = f"[{c['category']}]"
    title_line = f"  {c['title'].ljust(30)}{cat.rjust(16)}"
    line2 = f"  {c['driver_pos']}  {c['driver']}  ·  behind {c['ahead_pos']}  {c['ahead']}  ·  gap {c['gap']:.1f} s"
    line3 = f"  Projected loss: +{c['projected_loss_lap']:.2f} s / lap"
    line4 = f"  {c['train_info']}"

    box = (
        f"┌─────────────────────────────────────────────────┐\n"
        f"│{title_line.ljust(49)}│\n"
        f"│{line2.ljust(49)}│\n"
        f"│{line3.ljust(49)}│\n"
        f"│{line4.ljust(49)}│\n"
        f"└─────────────────────────────────────────────────┘"
    )
    return box


def render_png_card(c: dict, output_dir: str = "broadcast/rendered"):
    """Renders broadcast card as high-definition F1 television overlay graphic."""
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=180)
    fig.patch.set_facecolor("#15151E")
    ax.set_facecolor("#15151E")
    ax.axis("off")

    # Outer border & background
    card_bg = patches.FancyBboxPatch(
        (0.02, 0.04), 0.96, 0.92,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        edgecolor="#2D2D3D", facecolor="#1F1F2B", lw=1.5
    )
    ax.add_patch(card_bg)

    # Accent Red Top Stripe
    stripe = patches.Rectangle((0.05, 0.90), 0.90, 0.025, color="#E10600")
    ax.add_patch(stripe)

    # Header Row
    ax.text(0.06, 0.78, c["title"], color="#FFFFFF", fontsize=14, fontweight="heavy", va="center")
    ax.text(
        0.93, 0.78, f"[{c['category']}]",
        color=c["badge_color"], fontsize=10, fontweight="heavy", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#15151E", edgecolor=c["badge_color"], lw=1)
    )

    # Driver Sub-header
    ax.text(
        0.06, 0.58,
        f"{c['driver_pos']}  {c['driver']}   ·   BEHIND  {c['ahead_pos']}  {c['ahead']}   ·   GAP {c['gap']:.1f} s",
        color="#FFFFFF", fontsize=11, fontweight="bold", va="center"
    )

    # Projected Loss Delta
    ax.text(0.06, 0.38, "PROJECTED LOSS:", color="#949498", fontsize=10, va="center")
    loss_str = f"+{c['projected_loss_lap']:.2f} s / lap" if c['projected_loss_lap'] > 0 else "0.00 s (CLEAN AIR)"
    loss_col = "#E10600" if c['projected_loss_lap'] > 0.3 else ("#FF9900" if c['projected_loss_lap'] > 0 else "#10B981")
    ax.text(0.35, 0.38, loss_str, color=loss_col, fontsize=15, fontweight="heavy", va="center")

    # Detail / Train info
    ax.text(0.06, 0.20, c["train_info"], color="#949498", fontsize=9, va="center")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{c['id'].lower()}.png"
    plt.tight_layout()
    plt.savefig(file_path, facecolor="#15151E", edgecolor="none")
    plt.close()
    return str(file_path)


def generate_all_broadcast_cards(output_dir: str = "broadcast/rendered"):
    """Generates all cards both in ASCII text and PNG graphics."""
    ascii_cards = []
    png_paths = []
    for c in SAMPLE_BROADCAST_CARDS:
        ascii_box = render_ascii_card(c)
        ascii_cards.append(ascii_box)
        png_p = render_png_card(c, output_dir)
        png_paths.append(png_p)
    return ascii_cards, png_paths


if __name__ == "__main__":
    cards_txt, paths = generate_all_broadcast_cards()
    print(f"Generated {len(paths)} Broadcast PNG Cards.")
    print("\n--- SAMPLE ASCII BROADCAST CARD ---")
    print(cards_txt[0])
