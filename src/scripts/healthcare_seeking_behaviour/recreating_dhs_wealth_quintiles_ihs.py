"""
IHS5 (2019/20) wealth index, following the DHS Malawi 2010 SPSS syntax (Kiersten):
  - dichotomised categorical dummies (water, toilet own/shared, floor, wall, roof, cooking fuel)
  - assets, land (hectares x10, as in DHS hv245), members per sleeping room
  - mean-imputation of missings, PCA (1st component on correlation matrix, unweighted)
  - quintiles from the PERSON-weighted distribution of the score (hh weight x hh size)

Codes are taken from the IHS5 household questionnaire (modules F, F_1, L).
Variable NAMES are inferred from the questionnaire numbering (e.g. F40 -> hh_f40);
check anything flagged "VERIFY" against the data dictionary. Missing optional
variables are skipped and listed at the end rather than crashing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA = Path("/Users/rachelmurray-watson/Documents/HSB/MWI_2019_IHS-V_v06_M_CSV")
OUT = DATA / "wealth_index_output"
OUT.mkdir(exist_ok=True)

BASE_FILE = DATA / "hh_mod_a_filt.csv"
HOUSING_FILE = DATA / "HH_MOD_F.csv"
LAND_FILE = DATA / "HH_MOD_F1.csv"      # VERIFY: land roster (module F_1)
DURABLE_FILE = DATA / "HH_MOD_L.csv"
MEMBERS_FILE = DATA / "HH_MOD_B.csv"

# household id candidates (IHS5 usually has y4_hhid; case_id is the cross-section id)
HHID_CANDIDATES = ["y4_hhid", "hhid", "case_id"]
WEIGHT_CANDIDATES = ["hh_wgt", "hhweight", "weight"]   # VERIFY which weight you want


def read(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def pick(df, candidates, label, required=True):
    for c in candidates:
        if c.lower() in df.columns:
            return c.lower()
    if required:
        raise KeyError(f"No column for {label} among {candidates}. Columns: {list(df.columns)}")
    return None


skipped = []


def dummy(series, codes):
    """1 if value in codes, 0 otherwise; NaN stays NaN (so it can be mean-imputed like SPSS MEANSUB)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.isin(codes).astype(float).where(s.notna())


# ---------------------------------------------------------------- label -> code mapping
# The IHS5 CSVs hold value LABELS (text) for coded variables. These helpers turn them into the
# questionnaire codes used below. Columns that are already numeric pass through unchanged, so the
# same script works on the .dta (convert_categoricals=False). Rules are regexes matched against the
# lower-cased label, FIRST match wins (so specific patterns come before general ones).
WATER_RULES = [  # F40 / F21: main drinking water
    (r"piped into dwelling", 1), (r"piped into yard", 2), (r"piped to n", 17), (r"communal standpipe|standpipe", 3),
    (r"open well in yard", 4), (r"open public well", 5), (r"protected well in yard", 6),
    (r"protected public well", 7), (r"borehole", 8), (r"unprotected spring", 18), (r"protected spring", 9), (r"spring", 9),
    (r"river|stream", 10), (r"pond|lake", 11), (r"\bdam\b", 12), (r"rain", 13), (r"tanker|bowser", 14),
    (r"bottled", 15), (r"other", 16)]
TOILET_RULES = [  # F41
    (r"flush to piped|piped sewer", 1), (r"flush to septic|septic", 2), (r"flush to pit", 3),
    (r"flush to open drain|open drain", 4), (r"flush to d", 5), (r"ventilated|\bvip\b", 6),
    (r"without slab|open pit", 8), (r"with slab", 7), (r"composting", 9), (r"bucket", 10), (r"hanging", 11),
    (r"no facility|bush|field", 12), (r"other", 13)]
FLOOR_RULES = [  # F09
    (r"sand", 1), (r"smoothed mud|mud", 2), (r"smooth cement|cement", 3), (r"wood", 4), (r"tile", 5),
    (r"other", 6)]
WALL_RULES = [  # F07
    (r"grass", 1), (r"unfired|mud brick", 4), (r"burnt brick", 5), (r"compacted|yamdindo", 3), (r"mud|yomata", 2),
    (r"concrete", 6), (r"wood", 7), (r"iron", 8), (r"other", 9)]
ROOF_RULES = [  # F08
    (r"grass", 1), (r"iron", 2), (r"clay|tile", 3), (r"concrete", 4), (r"plastic", 5), (r"other", 6)]
FUEL_RULES = [  # F12
    (r"collected firewood", 1), (r"purchased firewood", 2), (r"grass|straw", 3), (r"paraffin", 4),
    (r"electric", 5), (r"\bgas\b", 6), (r"charcoal", 7), (r"crop residue", 8), (r"saw ?dust", 9),
    (r"animal", 10), (r"solar", 11), (r"other", 12)]
DURABLE_RULES = [  # module L item names -> item codes (only needed if the item column is text)
    (r"flash drive|micro cd", 5801), (r"^radio", 507), (r"upholstered|sofa", 522), (r"coffee table", 523),
    (r"^table", 503), (r"^chair", 504), (r"television|^tv\b", 509), (r"refrigerator|fridge", 514),
    (r"bicycle", 516), (r"motorcycle|scooter", 517), (r"mini-?bus", 519), (r"lorry", 520), (r"^car\b", 518),
    (r"lantern", 525)]
LANDUSE_RULES = [(r"residential", 1), (r"agricultur", 2), (r"pastoral", 3), (r"forest", 4),
                 (r"business|commercial", 5), (r"know", 6), (r"other", 7)]
UNIT_RULES = [(r"acre", 1), (r"hectare", 2), (r"square", 3), (r"other", 4)]


def _mostly_numeric(s):
    num = pd.to_numeric(s, errors="coerce")
    return num, bool(s.notna().any()) and num.notna().sum() >= 0.5 * s.notna().sum()


def recode(s, rules, name):
    """Numeric codes from either numeric input or text value labels (reports labels it cannot match)."""
    num, numeric = _mostly_numeric(s)
    if numeric:
        return num
    txt = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index)
    for pat, code in rules:
        m = txt.str.contains(pat, regex=True, na=False).astype(bool) & out.isna()
        out[m] = code
    unmatched = txt[out.isna() & txt.notna()].unique()
    if len(unmatched):
        print(f"[{name}] UNMATCHED labels (add a rule): {list(unmatched)[:10]}")
    return out


def yesno(s):
    """1 = yes, 2 = no (questionnaire coding), from numeric or text."""
    num, numeric = _mostly_numeric(s)
    if numeric:
        return num
    t = s.astype("string").str.strip().str.lower().fillna("")
    return pd.Series(np.where(t.str.startswith("yes"), 1, np.where(t.str.startswith("no"), 2, np.nan)),
                     index=s.index)


# ============================================================ 1. base + weight
base = read(BASE_FILE)
HHID = pick(base, HHID_CANDIDATES, "household id")
WGT = pick(base, WEIGHT_CANDIDATES, "household weight")
df = base[[HHID, WGT]].rename(columns={WGT: "hh_weight"}).drop_duplicates(HHID)
df["hh_weight"] = pd.to_numeric(df["hh_weight"], errors="coerce")
# DHS hv005 is stored x1e6; IHS weights are normally already in final form. Confirm before dividing.

# ============================================================ 2. housing (module F)
h = read(HOUSING_FILE)
h = h.rename(columns={pick(h, HHID_CANDIDATES, "household id (housing)"): HHID})
H = pd.DataFrame({HHID: h[HHID]})

water = pick(h, ["hh_f36"], "main drinking water")   # hh_f40 is the OTHER-season source (only ~350 non-null)
toilet = pick(h, ["hh_f41"], "toilet type F41")
# F41_4: share facility with other households? VERIFY name (hh_f41_4 / hh_f42 ...)
share = pick(h, ["hh_f41_4", "hh_f42", "hh_f41a"], "toilet shared F41_4", required=False)
floor = pick(h, ["hh_f09"], "floor F09")
wall = pick(h, ["hh_f07"], "wall F07")
roof = pick(h, ["hh_f08"], "roof F08")
rooms = pick(h, ["hh_f10"], "rooms F10")
fuel = pick(h, ["hh_f12"], "cooking fuel F12")
elec = pick(h, ["hh_f19"], "electricity F19")

# text labels -> questionnaire codes (no-op if the columns are already numeric)
h[water] = recode(h[water], WATER_RULES, "water F40")
h[toilet] = recode(h[toilet], TOILET_RULES, "toilet F41")
h[floor] = recode(h[floor], FLOOR_RULES, "floor F09")
h[wall] = recode(h[wall], WALL_RULES, "wall F07")
h[roof] = recode(h[roof], ROOF_RULES, "roof F08")
h[fuel] = recode(h[fuel], FUEL_RULES, "fuel F12")
h[elec] = yesno(h[elec])
if share:
    h[share] = yesno(h[share])

# --- water (F40): 1 piped dwelling, 2 piped yard, 17 piped to neighbour, 3 standpipe, 4/5 open well,
# 6/7 protected well, 8 borehole, 9 prot. spring, 18 unprot. spring, 10-12 surface, 13-16 other
H["h2oires"] = dummy(h[water], [1])
H["h2oyard"] = dummy(h[water], [2, 17])
H["h2opub"] = dummy(h[water], [3])
H["h2otube"] = dummy(h[water], [8])
H["h2opwell"] = dummy(h[water], [6, 7])
H["h2upwell"] = dummy(h[water], [4, 5])
H["h2spring"] = dummy(h[water], [9])
H["h2sprung"] = dummy(h[water], [18])
H["h2osurf"] = dummy(h[water], [10, 11, 12])
H["h2oother"] = dummy(h[water], [13, 14, 15, 16])

# --- toilet (F41): 1-5 flush, 6 VIP, 7 slab, 8 no slab, 9 composting, 10 bucket, 11 hanging, 12 none, 13 other
t = pd.to_numeric(h[toilet], errors="coerce")
if share:
    sh = pd.to_numeric(h[share], errors="coerce").eq(1)
else:
    skipped.append("toilet sharing variable not found -> all toilets treated as NOT shared")
    sh = pd.Series(False, index=h.index)
def tdummy(codes, shared):
    d = t.isin(codes) & (sh if shared else ~sh)
    return d.astype(float).where(t.notna())
H["flushs"], H["shflushs"] = tdummy([1, 2, 3, 4, 5], False), tdummy([1, 2, 3, 4, 5], True)
H["latvip"], H["shlatvip"] = tdummy([6, 9], False), tdummy([6, 9], True)        # VIP + composting (DHS 21/41)
H["latpits"], H["slatpits"] = tdummy([7], False), tdummy([7], True)
H["latpito"], H["slatpito"] = tdummy([8], False), tdummy([8], True)
H["latbush"] = dummy(h[toilet], [12])
H["latother"] = dummy(h[toilet], [10, 11, 13])

# --- electricity, rooms
H["electric"] = dummy(h[elec], [1])
H["rooms"] = pd.to_numeric(h[rooms], errors="coerce")

# --- floor (F09): 1 sand, 2 smoothed mud, 3 smooth cement, 4 wood, 5 tile, 6 other
H["dirtfloo"] = dummy(h[floor], [1, 2])
H["othfloo"] = dummy(h[floor], [6])
H["finfloo"] = dummy(h[floor], [3, 4, 5])
H["cemtfloo"] = dummy(h[floor], [3])

# --- wall (F07): 1 grass, 2 mud, 3 compacted earth, 4 unfired mud brick, 5 burnt brick, 6 concrete,
# 7 wood, 8 iron sheets, 9 other
H["grnwall"] = dummy(h[wall], [1])
H["dirtwall"] = dummy(h[wall], [2, 3])
H["stncwall"] = dummy(h[wall], [4])           # DHS: unburnt bricks
H["blckwall"] = dummy(h[wall], [5])           # burnt bricks
H["cmtwall"] = dummy(h[wall], [6])
H["othwall"] = dummy(h[wall], [7, 8, 9])

# --- roof (F08): 1 grass, 2 iron, 3 clay tiles, 4 concrete, 5 plastic, 6 other
H["natroof"] = dummy(h[roof], [1])
H["rudroof"] = dummy(h[roof], [2, 5, 6])
H["finroof"] = dummy(h[roof], [3, 4])

# --- cooking fuel (F12): 1 coll. wood, 2 purch. wood, 3 grass/straw, 4 paraffin, 5 electricity, 6 gas,
# 7 charcoal, 8 crop residue, 9 saw dust, 10 animal waste, 11 solar, 12 other
# (DHS put kerosene and "other" with wood)
H["cookelec"] = dummy(h[fuel], [5, 6, 11])
H["cookstrw"] = dummy(h[fuel], [3, 8, 10])
H["cookcoal"] = dummy(h[fuel], [7])
H["cookwood"] = dummy(h[fuel], [1, 2, 4, 9, 12])

# --- optional housing-module assets: bank account (F48), mobile phones (count), landline
bank = pick(h, ["hh_f48"], "bank account F48", required=False)
if bank: H["bank"] = dummy(yesno(h[bank]), [1])
else: skipped.append("bank (hh_f48)")
mph = pick(h, ["hh_f34", "hh_f36"], "number of working cell phones", required=False)  # VERIFY
if mph: H["mphone"] = (pd.to_numeric(h[mph], errors="coerce") > 0).astype(float).where(pd.to_numeric(h[mph], errors="coerce").notna())
else: skipped.append("mobile phone count (check F34/F35 variable name)")
lph = pick(h, ["hh_f31"], "MTL telephone in working condition", required=False)   # hh_f29 is the electricity-connection reason
if lph: H["lphone"] = dummy(yesno(h[lph]), [1])
else: skipped.append("landline (check variable name)")

df = df.merge(H.drop_duplicates(HHID), on=HHID, how="left", validate="one_to_one")

# ============================================================ 3. durables (module L)
d = read(DURABLE_FILE)
d = d.rename(columns={pick(d, HHID_CANDIDATES, "household id (durables)"): HHID})
# In this CSV: hh_l01 = owns item (Yes/No), hh_l02 = item NAME (text), hh_l03 = number owned,
# hh_l06 = purchased in last 12 months (NOT ownership).
own, item = "hh_l01", "hh_l02"
d[item] = recode(d[item], DURABLE_RULES, "durable item")     # item names -> module-L item codes
print(f"[durables] ownership={own} (yes share {yesno(d[own]).eq(1).mean():.2f}); "
      f"items recognised: {sorted(d[item].dropna().unique().astype(int).tolist())}")
d["_own"] = yesno(d[own]).eq(1)

DURABLES = {                      # item codes from module L
    "radio": [507, 5801], "tv": [509], "fridge": [514], "bicycle": [516], "motobk": [517],
    "car": [518, 519, 520],       # car, mini-bus, lorry (DHS hv212: car/truck)
    "sofaset": [522], "tbchair": None,  # table (503) AND chair (504), handled below
    "plamp": [525],               # paraffin lantern
}
owned = d[d["_own"]]
DUR = pd.DataFrame({HHID: df[HHID]}).set_index(HHID)
for name, codes in DURABLES.items():
    if codes is None:
        continue
    ids = set(owned.loc[owned[item].isin(codes), HHID])
    DUR[name] = DUR.index.isin(ids).astype(float)
tbl = set(owned.loc[owned[item] == 503, HHID]); chr_ = set(owned.loc[owned[item] == 504, HHID])
DUR["tbchair"] = DUR.index.isin(tbl & chr_).astype(float)
df = df.merge(DUR.reset_index(), on=HHID, how="left")
# households with no module-L rows are treated as owning nothing only if they are in the file at all
for c in DUR.columns:
    df[c] = df[c].fillna(0)
skipped.append("not in IHS5 module L / household module: watch, mattress (K405 is a purchase item), "
               "animal cart (module M, agriculture), koloboyi (merged into plamp)")

# ============================================================ 4. agricultural land (hectares x 10, as hv245)
try:
    land = read(LAND_FILE)
    land = land.rename(columns={pick(land, HHID_CANDIDATES, "household id (land)"): HHID})
    # In this CSV the land roster uses hh_f102* names: hh_f102a_1 = farmer-estimated area, hh_f102a_2 = its unit
    # (ACRE / HECTARE / SQUARE METERS). DHS hv245 is also farmer-reported, so the estimate is the faithful choice.
    est = pick(land, ["hh_f102a_1"], "farmer-estimated area", required=False)
    est_u = pick(land, ["hh_f102a_2"], "unit of estimate", required=False)
    gps = None
    # land-use column: a SHORT categorical column (<= 12 distinct labels) that contains 'agricultural'
    use = None
    for c in land.columns:
        if c.endswith("oth") or land[c].nunique() > 12:
            continue
        if land[c].astype(str).str.lower().str.contains("agricultur", na=False).any():
            use = c
            break
    if use:
        print(f"[land] use labels in {use}:", land[use].value_counts(dropna=False).to_dict())
    print(f"[land] use={use}, est={est}, est_unit={est_u}")
    if use:
        land = land[recode(land[use], LANDUSE_RULES, "land use").eq(2)]
    acres = pd.to_numeric(land[gps], errors="coerce") if gps else pd.Series(np.nan, index=land.index)
    if est and est_u:   # fall back to farmer estimate (units 1 acre, 2 ha, 3 m2)
        f = recode(land[est_u], UNIT_RULES, "land unit").map({1: 1.0, 2: 2.47105, 3: 1 / 4046.86})
        acres = acres.fillna(pd.to_numeric(land[est], errors="coerce") * f)
    land = land.assign(_ha=acres / 2.47105)
    L = land.groupby(HHID)["_ha"].sum(min_count=1).mul(10).rename("agland").reset_index()
    df = df.merge(L, on=HHID, how="left")
    df["agland"] = df["agland"].fillna(0)   # no ag land listed -> 0 (DHS: hv244 != 1 -> 0)
except Exception as e:                      # keep going; report
    skipped.append(f"agland ({e})")

# ============================================================ 5. household size, members per sleeping room
m = read(MEMBERS_FILE)
m = m.rename(columns={pick(m, HHID_CANDIDATES, "household id (roster)"): HHID})
# VERIFY: drop rows for people who left/died (roster B05_1 'still a member') if the data keep them.
hhsize = m.groupby(HHID).size().rename("hhsize").reset_index()
df = df.merge(hhsize, on=HHID, how="left")
df["rooms"] = df["rooms"].replace({0: 1})            # DHS: hv216 = 0 -> 1
df["memsleep"] = df["hhsize"] / df["rooms"]

# ============================================================ 6. PCA (unweighted, like the SPSS FACTOR step)
dummies = [c for c in ["h2oires", "h2oyard", "h2opub", "h2otube", "h2opwell", "h2upwell", "h2spring", "h2sprung",
                       "h2osurf", "h2oother", "flushs", "shflushs", "latvip", "shlatvip", "latpits", "slatpits",
                       "latpito", "slatpito", "latbush", "latother", "electric", "radio", "tv", "fridge", "bicycle",
                       "motobk", "car", "lphone", "mphone", "bank", "plamp", "sofaset", "tbchair",
                       "dirtfloo", "othfloo", "finfloo", "cemtfloo", "grnwall", "dirtwall", "stncwall", "blckwall",
                       "cmtwall", "othwall", "natroof", "rudroof", "finroof", "cookelec", "cookstrw", "cookcoal",
                       "cookwood"] if c in df.columns]
pca_vars = (["agland"] if "agland" in df.columns else []) + ["memsleep"] + dummies

df[pca_vars].isna().sum().to_csv(OUT / "missing_before_imputation.csv", header=["n_missing"])
for c in pca_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].mean())               # MISSING=MEANSUB

# SPSS drops nothing automatically, but a zero-variance column breaks standardisation: drop and report
const = [c for c in pca_vars if df[c].std(ddof=0) == 0]
if const:
    skipped.append(f"zero-variance variables dropped from PCA: {const}")
    pca_vars = [c for c in pca_vars if c not in const]

X = StandardScaler().fit_transform(df[pca_vars])
pca = PCA(n_components=1)
df["wealth_score"] = pca.fit_transform(X)[:, 0]
loadings = pd.Series(pca.components_[0], index=pca_vars, name="loading")
if loadings[[c for c in ["electric", "tv", "fridge", "car", "finfloo", "finroof"] if c in loadings]].mean() < 0:
    df["wealth_score"] *= -1; loadings *= -1
loadings.to_csv(OUT / "pca_loadings.csv")

# ============================================================ 7. person-weighted quintiles
df["person_weight"] = df["hh_weight"] * df["hhsize"]
ok = df["wealth_score"].notna() & df["person_weight"].gt(0)
s = df.loc[ok].sort_values("wealth_score")
cum = s["person_weight"].cumsum() / s["person_weight"].sum()
q = np.ceil(cum * 5 - 1e-12).clip(1, 5).astype(int)      # ties at a cutoff go to the lower group
df["wealth_quintile"] = pd.Series(q, index=s.index).reindex(df.index).astype("Int64")
cut = [s.loc[cum.le(p), "wealth_score"].max() for p in (0.2, 0.4, 0.6, 0.8)]
pd.Series(cut, index=["20%", "40%", "60%", "80%"], name="cutoff").to_csv(OUT / "wealth_quintile_cutoffs.csv")

# ============================================================ 8. means by quintile (SPSS MEANS) + outputs
def wmean(g, v):
    return np.average(g[v], weights=g["person_weight"])
summary = pd.DataFrame({v: df.dropna(subset=["wealth_quintile"]).groupby("wealth_quintile")
                        .apply(lambda g, v=v: wmean(g, v), include_groups=False) for v in pca_vars})
summary.to_csv(OUT / "means_by_wealth_quintile.csv")
df.to_csv(OUT / "household_wealth_index.csv", index=False)

print("PCA variables:", len(pca_vars), "| variance explained by PC1: %.1f%%" % (100 * pca.explained_variance_ratio_[0]))
print(loadings.sort_values().to_string())
print("Cutoffs:", cut)
print("Person-weighted share by quintile:\n", df.groupby("wealth_quintile")["person_weight"].sum() / df["person_weight"].sum())
print("\nSkipped / to check:")
for x in skipped: print(" -", x)
