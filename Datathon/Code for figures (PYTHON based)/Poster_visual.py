"""
Poster: Country-level weighted mean access attitudes, split by gender
3 radial donut charts — one per access variable
Each country arc split into 2 sub-wedges: outer = Men, inner = Women
Color coded light→dark by score, same palette per variable
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# ── NA codes ──────────────────────────────────────────────────────────────────
NA_CODES = {6, 7, 8, 9, 66, 77, 88, 99, 6666, 7777, 8888, 9999}

# ── ESS country lookup ────────────────────────────────────────────────────────
ESS_COUNTRY_NAMES = {
    "AT": "Austria",     "BE": "Belgium",     "BG": "Bulgaria",
    "CH": "Switzerland", "CY": "Cyprus",      "CZ": "Czechia",
    "DE": "Germany",     "DK": "Denmark",     "EE": "Estonia",
    "ES": "Spain",       "FI": "Finland",     "FR": "France",
    "GB": "UK",          "GR": "Greece",      "HR": "Croatia",
    "HU": "Hungary",     "IE": "Ireland",     "IL": "Israel",
    "IS": "Iceland",     "IT": "Italy",       "LT": "Lithuania",
    "LV": "Latvia",      "ME": "Montenegro",  "MK": "N. Macedonia",
    "NL": "Netherlands", "NO": "Norway",      "PL": "Poland",
    "PT": "Portugal",    "RS": "Serbia",      "RU": "Russia",
    "SE": "Sweden",      "SI": "Slovenia",    "SK": "Slovakia",
    "TR": "Turkey",      "UA": "Ukraine",     "XK": "Kosovo",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — replace simulation block with:
#   df = pd.read_csv("/path/to/ess1011_cronos3_withchild.csv")
# ══════════════════════════════════════════════════════════════════════════════
# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/bonnie/i4ng-datathon/r-project/data/ess1011_cronos3_withchild.csv")

print(f"Raw rows: {len(df):,}")
print(f"cntry dtype: {df['cntry'].dtype}")
print(f"cntry unique values:\n{sorted(df['cntry'].dropna().unique())}\n")

# ── Normalise cntry to clean string ──────────────────────────────────────────
# Handles numeric codes, float artefacts, and extra whitespace

df["cntry"] = (
    df["cntry"]
    .astype(str)           # coerce any numeric/factor encoding to string
    .str.strip()           # remove whitespace
    .str.upper()           # normalise case
    .replace("NAN", np.nan)
    .replace("", np.nan)
)

# ── NORMALISE gndr.x — coerce to int, keep only 1/2, everything else → NaN ───
df["gndr.x"] = pd.to_numeric(df["gndr.x"], errors="coerce")
df["gndr.x"] = df["gndr.x"].where(df["gndr.x"].isin([1, 2]), other=np.nan)

# ── FILTER ────────────────────────────────────────────────────────────────────
ACCESS_VARS = ["access_childcare", "access_parental_leave", "access_suitable_housing"]

df = df[
    df["w4weight"].notna() &
    df["cntry"].notna() &
    df["gndr.x"].notna()       # exclude NA gender
].copy()

for v in ACCESS_VARS:
    df[v] = pd.to_numeric(df[v], errors="coerce")
    df[v] = df[v].where(~df[v].isin(NA_CODES), other=np.nan)
    # if scale is reversed (1=easy, 4=hard) uncomment: df[v] = 5 - df[v]

# ── Derive country list ───────────────────────────────────────────────────────
CNTRY_CODES = sorted(df["cntry"].dropna().unique().tolist())
COUNTRIES   = {c: ESS_COUNTRY_NAMES.get(c, c) for c in CNTRY_CODES}

print(f"Analytic sample: n={len(df):,}  |  {len(CNTRY_CODES)} countries")
print(f"Gender breakdown:\n{df['gndr.x'].value_counts().to_string()}\n")


# ── WEIGHTED MEAN ─────────────────────────────────────────────────────────────
def wmean(series, weights):
    mask = series.notna()
    if mask.sum() == 0: return np.nan
    y, w = series[mask].values, weights[mask].values
    return np.sum(w * y) / np.sum(w)


# ── BUILD COUNTRY × GENDER SUMMARY ───────────────────────────────────────────
records = []
for c in CNTRY_CODES:
    for g, glabel in [(1, "Men"), (2, "Women")]:
        sub = df[(df["cntry"] == c) & (df["gndr.x"] == g)]
        rec = {"cntry": c, "country": COUNTRIES[c], "gndr": g, "gndr_label": glabel}
        for v in ACCESS_VARS:
            rec[v] = wmean(sub[v], sub["w4weight"])
        records.append(rec)

gdf = pd.DataFrame(records)

print("Weighted means by country × gender (childcare):")
print(gdf.pivot(index="country", columns="gndr_label",
                values="access_childcare").round(3).to_string())


# ─── DESIGN ───────────────────────────────────────────────────────────────────
BG         = "#0F1117"
TEXT_LIGHT = "#F0EDE8"
TEXT_MID   = "#B0ADA8"
RING_BG    = "#1E2130"

VAR_META = [
    {
        "var":    "access_childcare",
        "title":  "CHILDCARE",
        "colors": ["#D4EDFF", "#5BA8D4", "#1B5C8A", "#0A2E50"],
        "accent": "#5BA8D4",
    },
    {
        "var":    "access_parental_leave",
        "title":  "PARENTAL LEAVE",
        "colors": ["#D4F5E9", "#4DC9A0", "#1A7A5E", "#063D2E"],
        "accent": "#4DC9A0",
    },
    {
        "var":    "access_suitable_housing",
        "title":  "HOUSING",
        "colors": ["#FFE8CC", "#F0A050", "#A85A10", "#4A2200"],
        "accent": "#F0A050",
    },
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Georgia", "Times New Roman", "DejaVu Serif"],
    "text.color":  TEXT_LIGHT,
})


# ══════════════════════════════════════════════════════════════════════════════
# DRAW FUNCTION — gender-split radial chart
# ══════════════════════════════════════════════════════════════════════════════
def draw_radial_gender(ax, country_names, scores_m, scores_f, meta, vmin, vmax):
    """
    Hollow ring divided into N country slots.
    Each slot split angularly into:
      - Left half  = Men   (gndr=1)
      - Right half = Women (gndr=2)
    Both colored by their own weighted mean score.
    Label outside: country name, ♂ score / ♀ score.
    """
    cmap    = LinearSegmentedColormap.from_list("var", meta["colors"], N=256)
    n       = len(country_names)
    gap     = 1.5                        # Decreased distance between country slots
    arc     = (360 - n * gap) / n        # degrees per country slot
    sub_gap = 0.5                        # degrees between M/F sub-wedges

    # Radial bounds (Increased overall size)
    r_out   = 1.60
    r_in    = 0.90                       
    start   = 90                         # 12 o'clock

    ax.set_aspect("equal")
    # Expanded axes limits to accommodate the larger graph
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.axis("off")
    ax.set_facecolor(BG)

    def score_color(score):
        if pd.isna(score): return "#2A2A3A"
        t = np.clip((score - vmin) / (vmax - vmin), 0, 1)
        return cmap(t)

    for i, country in enumerate(country_names):
        theta1        = start - i * (arc + gap)
        theta2        = theta1 - arc
        theta_mid     = (theta1 + theta2) / 2
        theta_mid_rad = np.radians(theta_mid)

        sm = scores_m[i]
        sf = scores_f[i]

        # ── Background slots ──────────────────────────────────────────────
        ax.add_patch(mpatches.Wedge(
            (0,0), r_out, theta2, theta1, width=r_out - r_in,
            facecolor=RING_BG, edgecolor=BG, linewidth=1.2, zorder=1
        ))

        # ── Men — Left band (Higher angle) ────────────────────────────────
        ax.add_patch(mpatches.Wedge(
            (0,0), r_out, theta_mid + sub_gap/4, theta1 - sub_gap/2,
            width=r_out - r_in,
            facecolor=score_color(sm), edgecolor=BG, linewidth=1.0, zorder=2
        ))

        # ── Women — Right band (Lower angle) ──────────────────────────────
        ax.add_patch(mpatches.Wedge(
            (0,0), r_out, theta2 + sub_gap/2, theta_mid - sub_gap/4,
            width=r_out - r_in,
            facecolor=score_color(sf), edgecolor=BG, linewidth=1.0, zorder=2
        ))

        # ── Tick line outward ─────────────────────────────────────────────
        # Adjusted tick lengths for the larger radius
        r_tick0 = r_out + 0.04
        r_tick1 = r_out + 0.20
        r_text  = r_out + 0.32

        lx0 = r_tick0 * np.cos(theta_mid_rad)
        ly0 = r_tick0 * np.sin(theta_mid_rad)
        lx1 = r_tick1 * np.cos(theta_mid_rad)
        ly1 = r_tick1 * np.sin(theta_mid_rad)
        ax.plot([lx0, lx1], [ly0, ly1],
                color=meta["accent"], linewidth=0.7, alpha=0.5, zorder=3)

        # ── Label: country name + M/F scores ─────────────────────────────
        tx = r_text * np.cos(theta_mid_rad)
        ty = r_text * np.sin(theta_mid_rad)

        sm_str = f"{sm:.2f}" if not pd.isna(sm) else "—"
        sf_str = f"{sf:.2f}" if not pd.isna(sf) else "—"
        label  = f"{country}\nM {sm_str}  F {sf_str}"

        ax.text(tx, ty, label,
                ha="center", va="center",
                fontsize=9.0, color=TEXT_LIGHT,  # Increased font size
                linespacing=1.5, zorder=5)

    # ── Inner fill ────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((0,0), r_in - 0.01, color=BG, zorder=3))

    # ── Centre text ───────────────────────────────────────────────────────────
    ax.text(0,  0.30, meta["title"],
            ha="center", va="center",
            fontsize=15.0, fontweight="bold",    # Increased font size
            color=meta["accent"], zorder=6)
    ax.text(0, -0.02, "weighted mean",
            ha="center", va="center",
            fontsize=8.0, color=TEXT_MID, zorder=6)      # Increased font size
    ax.text(0, -0.20, "1 = very difficult  ·  4 = easy",
            ha="center", va="center",
            fontsize=7.5, color=TEXT_MID, style="italic", zorder=6) # Increased font size

    # ── Band legend (inner circle) ────────────────────────────────────────────
    for label, yoff in [
        ("M Men (Left)",   -0.45),
        ("F Women (Right)", -0.58),
    ]:
        ax.text(-0.18, yoff, label,
                ha="left", va="center",
                fontsize=7.0, color=TEXT_MID, zorder=6)  # Increased font size

    # ── Colour scale bar ──────────────────────────────────────────────────────
    bar_w, bar_h = 0.35, 0.045
    bar_x0, bar_y0 = -bar_w/2, -0.75
    n_steps = 100
    for si in range(n_steps):
        t = si / (n_steps - 1)
        ax.add_patch(mpatches.Rectangle(
            (bar_x0 + si * bar_w/n_steps, bar_y0),
            bar_w/n_steps + 0.001, bar_h,
            color=cmap(t), zorder=6, linewidth=0
        ))
    ax.add_patch(mpatches.FancyBboxPatch(
        (bar_x0, bar_y0), bar_w, bar_h,
        boxstyle="round,pad=0.004",
        linewidth=0.5, edgecolor=TEXT_MID,
        facecolor="none", zorder=7
    ))
    ax.text(bar_x0 - 0.05, bar_y0 + bar_h/2,
            f"{vmin:.1f}", ha="right", va="center",
            fontsize=9.0, color=TEXT_MID, zorder=7)      # Increased font size
    ax.text(bar_x0 + bar_w + 0.05, bar_y0 + bar_h/2,
            f"{vmax:.1f}", ha="left", va="center",
            fontsize=9.0, color=TEXT_MID, zorder=7)      # Increased font size
    
# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 11))
fig.patch.set_facecolor(BG)

fig.text(0.5, 0.975,
         "ACCESS TO FAMILY SERVICES ACROSS EUROPE",
         ha="center", va="top",
         fontsize=22, fontweight="bold",
         color=TEXT_LIGHT, fontfamily="serif")

fig.text(0.5, 0.915,
         f"{len(CNTRY_CODES)} countries  ·  Survey-weighted means  ·  "
         "Outer band = Men  ·  Inner band = Women  ·  "
         "Darker = easier access  ·  Scale 1–4",
         ha="center", va="top",
         fontsize=9.2, color=TEXT_MID,
         fontfamily="serif", style="italic")

# Global vmin/vmax for consistent colour scale across all panels
all_vals = gdf[ACCESS_VARS].values.flatten()
VMIN     = np.nanmin(all_vals)
VMAX     = np.nanmax(all_vals)

for col_idx, meta in enumerate(VAR_META):
    ax = fig.add_axes([0.03 + col_idx * 0.325, 0.02, 0.30, 0.86])
    ax.set_facecolor(BG)

    var = meta["var"]

    # Sort by average of M+F score so highest-access countries cluster together
    avg_score = gdf.groupby("country")[var].mean()
    sorted_countries = avg_score.sort_values(ascending=False).index.tolist()

    scores_m = [
        gdf[(gdf["country"] == c) & (gdf["gndr"] == 1)][var].values[0]
        for c in sorted_countries
    ]
    scores_f = [
        gdf[(gdf["country"] == c) & (gdf["gndr"] == 2)][var].values[0]
        for c in sorted_countries
    ]

    draw_radial_gender(ax, sorted_countries, scores_m, scores_f,
                       meta, VMIN, VMAX)

fig.text(0.5, 0.005,
         "Source: ESS10 / ESS11 / CRONOS 3  |  Weighted by w4weight  |  "
         "NA & invalid codes excluded  |  M = Men (gndr=1), F = Women (gndr=2)",
         ha="center", va="bottom",
         fontsize=7.5, color=TEXT_MID, style="italic")

fig.savefig("/Users/bonnie/i4ng-datathon/r-project/data/poster_3.png",
            dpi=220, bbox_inches="tight", facecolor=BG)
print("Poster saved.")
