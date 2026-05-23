import numpy as np
import pandas as pd
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# You can remove if you don’t use CARTOCOLORS anywhere
from palettable import cartocolors  # optional

# Load data
df = pd.read_csv("data-tv-series.csv")

# Ensure numeric types
df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")
df["end_year"]   = pd.to_numeric(df["end_year"],   errors="coerce")
df["stopped_year"] = pd.to_numeric(df["stopped_year"], errors="coerce")
df["ranking_score"] = pd.to_numeric(df["ranking_score"], errors="coerce")

# Broadcast duration
df["broadcast_years"] = df["end_year"] - df["start_year"]

# Separate stopped vs active (optional)
df_stopped = df.query("still_watching == 'no'")
df_active  = df.query("still_watching == 'yes'")

# --- "df_connect": line connecting start to stop (or end) ---
df_connect = df[["serie_name", "start_year", "end_year", "stopped_year", "still_watching"]].copy()

# Where the line ends visually:
# - if still_watching == "yes": end at `end_year`
# - if still_watching == "no": end at `stopped_year`
df_connect["visual_end"] = df_connect["end_year"]
df_connect.loc[df_connect["still_watching"] == "no", "visual_end"] = df_connect["stopped_year"]

# Dotted broadcast line: from stopped_year to end_year for stopped series
df_connect["dotted_start"] = df_connect["stopped_year"]
df_connect["dotted_end"]   = df_connect["end_year"]

# --- Sorting by first broadcast (oldest at top) ---
df_sorted = df.sort_values("start_year", ascending=True)

n_series = len(df_sorted)
ypos = np.arange(n_series)

df_sorted["serie_name"] = pd.Categorical(
    df_sorted["serie_name"],
    ordered=True,
    categories=df_sorted["serie_name"].tolist()
)

df_connect["serie_name"] = df_connect["serie_name"].astype("category")
df_connect["serie_name"].cat.set_categories(df_sorted["serie_name"].dtype.categories.tolist())
df_connect = df_connect.sort_values("serie_name")

# --- Color mapping ---
min_score = df["ranking_score"].min()
max_score = df["ranking_score"].max()

BAD = "#e74c3c"
NEUTRAL = "#f39c12"
GOOD = "#27ae60"

colormap = mc.LinearSegmentedColormap.from_list("series_rating", [BAD, NEUTRAL, GOOD], N=256)
norm = mc.Normalize(vmin=min_score, vmax=max_score)

# ScalarMappable for colorbar (important!)
sm = ScalarMappable(norm=norm, cmap=colormap)
sm.set_array([])

# --- Colors for layout   ---
# You can define these explicitly if you like
GREY94 = "#f0f0f0"
GREY75 = "#bfbfbf"
GREY65 = "#a6a6a6"
GREY55 = "#8c8c8c"
GREY50 = "#7f7f7f"
GREY40 = "#666666"
LIGHT_BLUE = "#b4d1d2"
DARK_BLUE = "#242c3c"
BLUE = "#4a5a7b"
WHITE = "#FFFCFC" # technically not pure white

# =====================
# PLOT
# =====================
fig, ax = plt.subplots(figsize=(14, 8))

ax.set_yticks(ypos)
ax.set_yticklabels(df_sorted["serie_name"])

ax.set_xlim(2010, 2027)
ax.set_xlabel("Year")

# --- Main loop: draw lines, then dots ---
for i, row in df_connect.iterrows():
    idx = df_sorted[df_sorted["serie_name"] == row["serie_name"]].index[0]
    y = ypos[idx]

    score = df_sorted.loc[idx, "ranking_score"]
    color = colormap(norm(score))

    # Solid line: what you actually watched
    ax.hlines(
        y=y,
        xmin=row["start_year"],
        xmax=row["visual_end"],
        color=color,
        lw=4,
    )

    # Dotted line: broadcast period after you stopped
    if row["still_watching"] == "no":
        ax.hlines(
            y=y,
            xmin=row["dotted_start"],
            xmax=row["dotted_end"],
            color="grey",
            ls=":",
            lw=2,
        )

    # Dots: start and end/stop
    ax.scatter(
        row["start_year"],
        y,
        s=100,
        color="white",
        edgecolor=color,
        lw=2
    )

    ax.scatter(
        row["visual_end"],
        y,
        s=100,
        color="black" if row["still_watching"] == "no" else "white",
        edgecolor=color,
        lw=2
    )

# --- Labels on the left side (like MK64 track names) ---
for i, row in df_connect.iterrows():
    idx = df_sorted[df_sorted["serie_name"] == row["serie_name"]].index[0]
    y = ypos[idx]

    ax.text(
        row["start_year"] - 0.5,   # a bit to the left
        y,
        row["serie_name"],
        ha="right",
        va="center",
        size=14,
        color="black",
        fontname="Arial"
    )

# --- Layout: hide spines, ticks, etc. ---
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(True)

# You can keep y‑labels *or* text on the left, but not both
ax.yaxis.set_visible(False)

ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True, length=0)
xticks = list(range(2012, 2026, 2))
ax.set_xlim(2011, 2026)
ax.set_xticks(xticks)
ax.set_xticklabels(
    [str(x) for x in xticks],
    fontname="Arial",
    color=GREY40,
    size=9
)

ax.set_facecolor(WHITE)

for xtick in xticks:
    ax.axvline(xtick, color=GREY94, zorder=0)

# --- Only one call to plt.axis for vertical space; remove the duplicated one ---
x0, x1, y0, y1 = plt.axis()
plt.axis((x0, x1, y0 - 0.5, y1 + 0.5))

# ==============
# Legends & Titles
# ==============

# Colorbar (MK64‑style)
cbaxes = inset_axes(
    ax,
    width="0.8%",
    height="44%",
    loc=3,
    bbox_to_anchor=(0.025, 0., 1, 1),
    bbox_transform=ax.transAxes
)

cb = fig.colorbar(
    sm,
    cax=cbaxes,
    ticks=[min_score, (min_score + max_score)/2, max_score]
)

cb.outline.set_visible(False)
cb.set_label(
    "Series Ranking (1–10)",
    labelpad=-45,
    color=GREY40,
    size=10,
    fontname="Arial"
)
cb.ax.yaxis.set_tick_params(size=0)
cb.ax.yaxis.set_ticklabels(
    [min_score, (min_score + max_score)/2, max_score],
    fontname="Arial",
    color=GREY40,
    size=10
)

# ==============================================================
# Titles & footer
plt.suptitle(
    "My TV Series Watch-History Timeline",
    fontsize=14,
    fontname="Arial",
    weight="bold",
    x=0.45,
    y=0.98
)

ax.set_title(
    "Shows ordered by year of first broadcast (oldest at bottom). "
    "Solid lines show watched period; dotted lines show broadcast period I did not watch.\n"
    "Color represents personal ranking (1-10).",
    loc="center",
    color=GREY40,
    fontname="Arial",
    fontsize=9,
    pad=20
)

fig.text(
    0.8,
    0.05,
    "Visualization: Emeline Simon • Data: my_series.csv",
    fontname="Arial",
    fontsize=11,
    color=GREY94,
    ha="center"
)

# Background and save
fig.patch.set_facecolor(WHITE)

plt.savefig(
    "my_series_timeline.png",
    facecolor=WHITE,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3
)

# Optional: show if you’re in an interactive environment
# plt.show()