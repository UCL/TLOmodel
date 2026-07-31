"""
plot_district_indicator_heatmap.py

Reads district_burden_{indicator}.csv files and plots a district × indicator
heatmap of the two-model deficit (%). Diverging colormap centered on 0.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/"

INDICATOR_ORDER = [
    "vmmc_first_visits",
    "opd_attendance",
    "ipd_total_admissions",
    "measles1_under1",
    "fully_immunised_under1",
    "fp_total_clients",
    "pnc_mother_checked_48h",
    "penta3_under1",
    "live_births_total",
    "bcg_under1",
    "pnc_within_2wks",
]

INDICATOR_LABELS = {
    "fp_total_clients":         "FP Total Clients",
    "opd_attendance":           "OPD Attendance",
    "ipd_total_admissions":     "IPD Total Admissions",
    "vmmc_first_visits":        "VMMC First Visits",
    "pnc_mother_checked_48h":   "PNC Mother <48h",
    "bcg_under1":               "BCG Under-1",
    "penta3_under1":            "Penta3 Under-1",
    "measles1_under1":          "Measles 1st Dose Under-1",
    "fully_immunised_under1":   "Fully Immunised Under-1",
    "pnc_within_2wks":          "PNC Within 2 Weeks",
    "live_births_total":        "Live Births Total",
}

# Build long-format table.
rows = []
for ind in INDICATOR_ORDER:
    path = f"{OUT_DIR}district_burden_{ind}.csv"
    if not os.path.exists(path):
        print(f"  [{ind}] no district CSV — skipping")
        continue
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append({
            "district":  r["Dist"],
            "indicator": ind,
            # Flip sign so positive = services lost to heat.
            "services_lost_pct": -r["deficit_pct"],
        })

wide = (pd.DataFrame(rows)
        .pivot(index="district", columns="indicator", values="services_lost_pct")
        .reindex(columns=INDICATOR_ORDER))

# Order districts north → south (rough Malawi latitude order).
DISTRICT_ORDER = [
    "Chitipa", "Karonga", "Likoma", "Rumphi", "Mzimba", "Nkhata Bay",
    "Kasungu", "Nkhotakota", "Ntchisi", "Dowa", "Salima", "Lilongwe",
    "Mchinji", "Dedza", "Ntcheu", "Mangochi", "Balaka", "Machinga",
    "Zomba", "Chiradzulu", "Blantyre", "Mwanza", "Neno", "Phalombe",
    "Mulanje", "Thyolo", "Chikwawa", "Nsanje",
    "Mzuzu City", "Lilongwe City", "Blantyre City", "Zomba City",
]
wide = wide.reindex([d for d in DISTRICT_ORDER if d in wide.index])

# Symmetric limits so 0 is white.
vabs = float(np.nanpercentile(np.abs(wide.values), 95))

fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(wide) + 2)))
im = ax.imshow(wide.values, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

ax.set_xticks(range(len(wide.columns)))
ax.set_xticklabels(
    [INDICATOR_LABELS.get(c, c) for c in wide.columns],
    rotation=40, ha="right", fontsize=9,
)
ax.set_yticks(range(len(wide.index)))
ax.set_yticklabels(wide.index, fontsize=8)

# Annotate cells.
for i in range(len(wide.index)):
    for j in range(len(wide.columns)):
        v = wide.values[i, j]
        if pd.notna(v) and abs(v) >= 0.05:   # skip near-zero to reduce clutter
            ax.text(j, i, f"{v:+.1f}",
                    ha="center", va="center",
                    fontsize=6.5, color="black" if abs(v) < vabs*0.7 else "white")

ax.set_title(
    "District × indicator heat-attributable service loss (%)\n"
    "Positive = services lost, negative = services gained under heat",
    fontsize=11, fontweight="bold",
)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
cbar.set_label("% services lost to heat", fontsize=9)

# Separator between rural districts and cities.
n_rural = sum(1 for d in wide.index if "City" not in d)
if n_rural < len(wide.index):
    ax.axhline(n_rural - 0.5, color="black", lw=1.2, linestyle="--")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}district_indicator_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_DIR}district_indicator_heatmap.png")
