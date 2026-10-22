"""
Build the person-level healthcare-seeking dataset from IHS5 (2019/20) for a nested logit.

One row per person-illness episode (illness/injury in the past 2 weeks, module D).
Outputs hsb_analysis.csv with:
  choice            none / other / gov / cham / private   (see CHOICE_MAP)
  covariates        age, sex, symptom group, chronic, urban, region, wealth quintile, severity proxies...
  cost_<alt>        PREDICTED log(1+MWK) cost of each alternative (cell means among users; see section 6)
  dist_<alt>        km to nearest facility of that type, if a facility file is supplied (see section 7)

Variable NAMES in module D are inferred from the questionnaire and may differ in the release you hold.
Every candidate list below is tried in order; if none match, the script stops and prints the columns of
that file so you can fix the list. Anything marked VERIFY needs checking against the data dictionary.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("HSB_DATA", "/Users/rachelmurray-watson/Documents/HSB/MWI_2019_IHS-V_v06_M_CSV"))
OUT = DATA / "hsb_output"
OUT.mkdir(exist_ok=True)

WEALTH_FILE = DATA / "wealth_index_output" / "household_wealth_index.csv"   # from your PCA script
BASE_FILE = DATA / "hh_mod_a_filt.csv"          # region, district, reside, ea_id, weights
ROSTER_FILE = DATA / "HH_MOD_B.csv"             # sex, age
HEALTH_FILE = DATA / "HH_MOD_D.csv"             # illness, action, cost, chronic
EDUC_FILE = DATA / "HH_MOD_C.csv"               # optional: education
SHOCK_FILE = DATA / "HH_MOD_U.csv"              # optional: drought/flood shocks
HOUSING_FILE = DATA / "HH_MOD_F.csv"            # optional: bank account (F48)
GEO_FILE = DATA / "HouseholdGeovariables_IHS5.csv"  # optional: ea_lat_mod / ea_lon_mod (modified GPS), rainfall

# Optional external facility list (e.g. geocoded Malawi Health Facility Register / MHFR).
# CSV with columns: lat, lon, type  where type in {"gov", "cham", "private"}. Leave None to skip distances.
FACILITY_FILE = None   # Path("/path/to/mhfr_geocoded.csv")

HHID_C = ["y4_hhid", "hhid", "case_id"]
PID_C = ["pid", "id_code", "ind_id"]

# ---- candidate variable names (module D) -- VERIFY ----------------------------------------------
V = {
    "ill_any":   ["hh_d03", "hh_d04", "hh_d00"],          # illness/injury past 2 weeks (1=yes)       D01/D03
    "ill1":      ["hh_d05a", "hh_d05_1", "hh_d04a"],      # illness 1 code (1-32)                       D04
    "ill2":      ["hh_d05b", "hh_d05_2", "hh_d04b"],
    "diag1":     ["hh_d06a", "hh_d06_1", "hh_d05a_diag"], # who diagnosed illness 1                     D05
    "act1":      ["hh_d07a", "hh_d07_1", "hh_d06a_act"],  # action taken, problem 1 (1-13)              D06
    "act2":      ["hh_d07b", "hh_d07_2"],
    "days_stop": ["hh_d08", "hh_d07"],                    # days normal activity stopped                D07
    "days_care": ["hh_d09", "hh_d08b"],                   # days someone else stopped to care for you   D08
    "cost4w":    ["hh_d14", "hh_d15"],                    # spend past 4 wks, all illness/injury        VERIFY which Dxx
    "insur":     ["hh_d10"],                              # spend on medical insurance past 12m (>0 = insured)  D10 VERIFY
    "inpat":     ["hh_d18", "hh_d19"],                    # hospitalised/overnight past 12m
    "chronic":   ["hh_d33", "hh_d34"],                    # chronic illness yes/no                      D33
    "diff_see":  ["hh_d20"], "diff_hear": ["hh_d21"], "diff_walk": ["hh_d22"],
    "diff_mem":  ["hh_d23"], "diff_care": ["hh_d24"], "diff_speak": ["hh_d25"],
}

# ---- outcome definition -------------------------------------------------------------------------
# D06 action codes: 1 nothing/not serious, 2 nothing/no money, 3 stock medicine, 4 known remedies,
# 5 govt facility, 6 church/mission (CHAM), 7 private facility, 8 village clinic/HSA, 9 pharmacy,
# 10 grocery, 11 traditional healer, 12 faith healer, 13 other.


CHOICE_MAP = {1:"none", 2:"none", 3:"none", 4:"none",
              5:"gov", 8:"gov", 6:"cham",
              7:"none", 9:"none", 10:"none", 11:"none", 12:"none", 13:"none"}


# precedence when several actions are reported (highest wins): formal facility > other > none
RANK = {"none": 0, "gov": 1, "cham": 1}
# ---- symptom groups from D04 codes ---------------------------------------------------------------
SYMPTOM_GROUPS = {
    "fever_malaria": [1],
    "gi": [2, 3, 4],
    "resp": [5, 6, 7, 8, 9, 30],
    "injury": [24, 25, 26, 27],
    "pain_headache": [10, 11, 16, 31],
    "skin_eye_ear_dental": [12, 13, 14, 15],
    "pregnancy": [28],
    "other": [17, 18, 19, 20, 21, 22, 23, 29, 32],
}


def read(p):
    df = pd.read_csv(p, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def pick(df, cands, label, required=True, src=""):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        sys.exit(f"[stop] no column for '{label}' among {cands} in {src}.\nColumns: {sorted(df.columns)}")
    return None


def hhkey(df, src):
    """Standardise the household id. Tries each candidate id column (y4_hhid / hhid / case_id) against the matching column in
    the base file (cast to stripped str, so int/str mismatches are harmless). If the file uses a different id from HH but the
    base file carries that id too (e.g. geovariables keyed by case_id), it is translated to HH through the base file."""
    cands = [c for c in HHID_C if c in df.columns]
    if not cands:
        sys.exit(f"[stop] no household id column among {HHID_C} in {src}. Columns: {sorted(df.columns)}")
    best, mapping, ov = None, None, {}
    for c in cands:
        if c in base.columns:
            key = base[c].astype(str).str.strip()
            ov[c] = len(set(df[c].astype(str).str.strip()) & set(key))
            if best is None or ov[c] > ov[best]:
                best = c
                mapping = None if c == HH else dict(zip(key, base[HH]))
    if best is None or ov[best] == 0:
        sys.exit(f"[stop] cannot link '{src}' to the base file. Candidate id columns in {src}: {cands} (overlaps with base: {ov}). "
                 f"Base id columns: {[c for c in HHID_C if c in base.columns]}.\nColumns in {src}: {sorted(df.columns)}")
    df = df.drop(columns=[c for c in cands if c != best]).rename(columns={best: HH})
    df[HH] = df[HH].astype(str).str.strip()
    if mapping is not None:
        df[HH] = df[HH].map(mapping)
        df = df[df[HH].notna()]
    return df


def pidnorm(x):
    """person id as a clean str whether stored as int, float or str"""
    n = pd.to_numeric(x, errors="coerce")
    return np.where(n.notna(), n.astype("Int64").astype(str), x.astype(str).str.strip())


def num(s):
    return pd.to_numeric(s, errors="coerce")


notes = []

# ============================================================ 1. household info
base = read(BASE_FILE)
HH = pick(base, HHID_C, "household id", src="base")
base[HH] = base[HH].astype(str).str.strip()
BASE_IDS = set(base[HH])
reg = pick(base, ["region"], "region", src="base")
res = pick(base, ["reside", "urban", "rural"], "urban/rural", src="base")
ea = pick(base, ["ea_id", "ea"], "EA id", required=False, src="base")
wgt = pick(base, ["hh_wgt", "hhweight", "weight"], "household weight", required=False, src="base")
hh = base[[HH] + [c for c in (reg, res, ea, wgt) if c]].drop_duplicates(HH).rename(
    columns={reg: "region", res: "reside", **({ea: "ea_id"} if ea else {}), **({wgt: "hh_weight"} if wgt else {})})
hh["urban"] = (num(hh["reside"]) == 1).astype(int)       # VERIFY coding: IHS reside 1=urban, 2=rural
hh["region"] = num(hh["region"]).map({1: "North", 2: "Central", 3: "South"})
wealth = hhkey(read(WEALTH_FILE), "wealth")[[HH, "wealth_quintile"]]
hh = hh.merge(wealth, on=HH, how="left")

# ============================================================ 2. roster: sex, age
r = read(ROSTER_FILE)
r = hhkey(r, "roster")
PID = pick(r, PID_C, "person id", src="roster")
sexc = pick(r, ["hh_b03"], "sex", src="roster")
agec = pick(r, ["hh_b05a"], "age (years)", src="roster")
relc = pick(r, ["hh_b04"], "relationship to head", required=False, src="roster")
r[PID] = pidnorm(r[PID])
r = r[[HH, PID, sexc, agec] + ([relc] if relc else [])].rename(columns={PID: "pid", sexc: "sex_raw", agec: "age"})
r["female"] = (num(r["sex_raw"]) == 2).astype(int)
r["age"] = num(r["age"])
# household composition from the roster (VERIFY: includes people who have left/died if the file keeps them)
comp = r.groupby(HH).agg(hhsize=("pid", "size"), n_u5=("age", lambda a: int((a < 5).sum()))).reset_index()
comp["log_hhsize"] = np.log(comp["hhsize"])
comp["has_u5"] = (comp["n_u5"] > 0).astype(int)
if relc:
    head = r[num(r[relc]) == 1].drop_duplicates(HH)[[HH, "pid", "female"]].rename(columns={"pid": "head_pid", "female": "head_female"})
else:
    head = pd.DataFrame(columns=[HH, "head_pid", "head_female"])
    notes.append("head of household: hh_b04 not found -> head_female / head_educ skipped")
r = r.drop(columns=[c for c in (relc,) if c])

# ============================================================ 3. health module -> episodes
d = read(HEALTH_FILE)
d = hhkey(d, "health")
dpid = pick(d, PID_C, "person id", src="health")
d = d.rename(columns={dpid: "pid"})
d["pid"] = pidnorm(d["pid"])
cols = {k: pick(d, v, k, required=k in ("ill_any", "ill1", "act1"), src="health") for k, v in V.items()}
d = d[d[cols["ill_any"]].pipe(num).eq(1)].copy()
ep = d[[HH, "pid"]].copy()
for k in ("ill1", "act1", "act2", "days_stop", "days_care", "cost4w", "chronic", "inpat", "diag1", "insur",
          "diff_see", "diff_hear", "diff_walk", "diff_mem", "diff_care", "diff_speak"):
    if cols.get(k):
        ep[k] = num(d[cols[k]])
    else:
        notes.append(f"{k}: column not found, skipped")

# --- outcome: highest-ranked of action1/action2 (both refer to the two reported problems)
def classify(a):
    return CHOICE_MAP.get(int(a)) if pd.notna(a) else None
ch1 = ep["act1"].map(classify)
ch2 = ep["act2"].map(classify) if "act2" in ep else pd.Series(None, index=ep.index)
ep["choice"] = [max([c for c in (a, b) if c], key=lambda c: RANK[c], default=None) for a, b in zip(ch1, ch2)]
ep["any_care"] = ep["choice"].isin(["gov", "cham", "private", "other"]).astype(float).where(ep["choice"].notna())

# --- symptom group from the first-listed illness
code2grp = {c: g for g, cs in SYMPTOM_GROUPS.items() for c in cs}
ep["symptom"] = ep["ill1"].map(code2grp).fillna("other")

# --- chronic (D33): 1=yes. Coded 0 if answered 2; NaN if missing
if "chronic" in ep:
    ep["chronic"] = (ep["chronic"] == 1).astype(int)
# --- functional difficulty: any domain 'a lot' (3) or 'cannot' (4)
dif = [c for c in ep.columns if c.startswith("diff_")]
if dif:
    ep["disability"] = (ep[dif] >= 3).any(axis=1).astype(int)
    ep = ep.drop(columns=dif)

# ============================================================ 4. merge person + household
df = ep.merge(r, on=[HH, "pid"], how="left").merge(hh, on=HH, how="left").merge(comp, on=HH, how="left")
if len(head):
    df = df.merge(head[[HH, "head_female", "head_pid"]], on=HH, how="left")

# interview month (rainy season Nov-Apr flag). VERIFY the date variable name in the base file.
dt = pick(base, ["hh_a37", "hh_a23", "interview_date", "hh_a37_date", "hh_a23b"], "interview date", required=False, src="base")
if dt:
    mth = pd.to_datetime(base[dt], errors="coerce", dayfirst=False)
    base_m = pd.DataFrame({HH: base[HH], "int_month": mth.dt.month}).drop_duplicates(HH)
    df = df.merge(base_m, on=HH, how="left")
    df["rainy_season"] = df["int_month"].isin([11, 12, 1, 2, 3, 4]).astype(float).where(df["int_month"].notna())
else:
    notes.append("interview month: date column not found (set candidates in section 4) -> rainy_season skipped")

# bank account (F48) from the housing module
try:
    f_ = hhkey(read(HOUSING_FILE), "housing")
    bk = pick(f_, ["hh_f48"], "bank account", required=False, src="housing")
    if bk:
        df = df.merge(f_[[HH, bk]].drop_duplicates(HH).assign(bank=lambda x: (num(x[bk]) == 1).astype(int))[[HH, "bank"]], on=HH, how="left")
except Exception as e:
    notes.append(f"bank skipped ({e})")

# rainfall from geovariables: season total vs long-term average -> anomaly. VERIFY names in your geovariables file
geo = None
if GEO_FILE.exists():
    geo = hhkey(read(GEO_FILE), "geo")
    rs = pick(geo, ["h2019_tot", "h2018_tot"], "season rainfall total", required=False, src="geo")
    rl = pick(geo, ["anntot_avg"], "long-term annual rainfall", required=False, src="geo")
    if rs and rl:
        geo["rain_anom"] = (num(geo[rs]) - num(geo[rl])) / num(geo[rl])
        df = df.merge(geo[[HH, "rain_anom"]].drop_duplicates(HH), on=HH, how="left")
    else:
        notes.append("rain_anom skipped: rainfall columns not found in geovariables")
else:
    notes.append("geovariables file not found -> rain_anom skipped")
df["age_grp"] = pd.cut(df["age"], [-1, 4, 14, 34, 49, 200], labels=["0-4", "5-14", "15-34", "35-49", "50+"])

# optional: education of the household head (C09 highest qualification: 1 none, 2 PSLC, 3 JCE, 4 MSCE+, 5+ post-secondary)
# head's education is used for everyone so children have a value too.
try:
    c_ = hhkey(read(EDUC_FILE), "educ"); c_ = c_.rename(columns={pick(c_, PID_C, "pid", src="educ"): "pid"}); c_["pid"] = pidnorm(c_["pid"])
    q = pick(c_, ["hh_c09"], "highest qualification", required=False, src="educ")
    if q and "head_pid" in df:
        hq = c_[[HH, "pid", q]].rename(columns={"pid": "head_pid", q: "hq"})
        df = df.merge(hq, on=[HH, "head_pid"], how="left")
        df["head_educ"] = pd.cut(num(df["hq"]), [0, 1, 2, 3, 99], labels=["none", "primary", "jce", "msce_plus"]).astype(object)
        df = df.drop(columns=["hq"])
except Exception as e:
    notes.append(f"head education skipped ({e})")
try:
    u = hhkey(read(SHOCK_FILE), "shocks")
    code = pick(u, ["hh_u0a", "hh_u00"], "shock code", required=False, src="shocks")
    aff = pick(u, ["hh_u01"], "shock affected", required=False, src="shocks")
    if code and aff:
        u["climate"] = num(u[code]).isin([101, 1101, 102, 1102]) & num(u[aff]).eq(1)   # drought, irregular rains, floods, landslides
        df = df.merge(u.groupby(HH)["climate"].max().astype(int).rename("climate_shock").reset_index(), on=HH, how="left")
except Exception as e:
    notes.append(f"shocks skipped ({e})")

# ============================================================ 5. sample
df["child"] = (df["age"] < 15).astype(int)
if "insur" in df:
    df["insured"] = (df["insur"].fillna(0) > 0).astype(int)
if "days_stop" in df:
    df["days_stop"] = df["days_stop"].where(df["days_stop"] <= 14)      # 2-week window: >14 is an error / DK code
n0 = len(df)
df = df[df["choice"].notna() & df["age"].notna() & df["wealth_quintile"].notna() & df["region"].notna()]
notes.append(f"episodes with usable outcome/covariates: {len(df)} of {n0} illness episodes")

# ============================================================ 6. predicted cost per alternative
# Cost is only observed for the alternative a person chose, and D14-type spend is a total over ALL illnesses in 4 weeks
# (not facility-specific). We therefore build a *predicted* cost for every alternative = mean log(1+MWK) among people who
# chose it, within region x urban cells (fallback: alternative-wide mean). Predicted values are used for ALL alts incl.
# the chosen one so observed and imputed costs are not mixed. 'none' costs 0 by construction.
ALTS = ["none", "other", "gov", "cham", "private"]
if "cost4w" in df:
    df["lcost"] = np.log1p(df["cost4w"].where(df["cost4w"] < 9e5))   # 999999 / 99999 are DK codes: set NaN
    for a in ALTS:
        if a == "none":
            df["cost_none"] = 0.0
            continue
        users = df[(df["choice"] == a) & df["lcost"].notna()]
        cell = users.groupby(["region", "urban"])["lcost"].mean()
        df[f"cost_{a}"] = [cell.get((rg, ur), users["lcost"].mean()) for rg, ur in zip(df["region"], df["urban"])]
    notes.append("cost_* are cell-mean predictions (region x urban); identification comes only from that variation. "
                 "Consider facility-type x district means if n allows.")
else:
    notes.append("no cost column -> cost_* not created")

# ============================================================ 7. distance to nearest facility by type
# if FACILITY_FILE is not None and geo is not None:
#     from scipy.spatial import cKDTree
#     la = pick(geo, ["ea_lat_mod", "lat_modified", "latitude"], "lat", src="geo")
#     lo = pick(geo, ["ea_lon_mod", "lon_modified", "longitude"], "lon", src="geo")
#     g = geo[[HH, la, lo]].rename(columns={la: "lat", lo: "lon"}).drop_duplicates(HH)
#     df = df.merge(g, on=HH, how="left")
#     fac = read(FACILITY_FILE)
#     xyz = lambda lat, lon: np.c_[np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
#                                  np.cos(np.radians(lat)) * np.sin(np.radians(lon)), np.sin(np.radians(lat))]
#     P = xyz(df["lat"].values, df["lon"].values)
#     for a in ("gov", "cham", "private"):
#         f = fac[fac["type"] == a]
#         dist, _ = cKDTree(xyz(f["lat"].values, f["lon"].values)).query(np.nan_to_num(P))
#         df[f"dist_{a}"] = np.where(df["lat"].notna(), 2 * 6371 * np.arcsin(dist / 2), np.nan)   # chord -> great-circle km
#     df["dist_none"] = 0.0
#     df["dist_other"] = 0.0
#     notes.append("distances are EA-centroid (IHS5 modified coords, displaced up to 2 km urban / 5 km rural) to nearest facility "
#                  "of each type; straight-line, not travel time.")
# else:
#     notes.append("distance NOT built: set FACILITY_FILE (geocoded facility list with lat, lon, type) and check GEO_FILE.")

df.to_csv(OUT / "hsb_analysis.csv", index=False)
print(df["choice"].value_counts(normalize=True).round(3).to_string())
print(f"\nN episodes = {len(df)}  | children = {df['child'].mean():.2f}")
print("\nNotes / to check:")
for n in notes:
    print(" -", n)
