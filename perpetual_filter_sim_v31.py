
"""
Perpetual Metabolic Filter Simulation (V31)
===========================================

A self-contained, auditable Python module implementing the simulation framework
described in "The Perpetual Filter Theory / Metabolic Society" hypothesis.

Key features
------------
- Five-century horizon (1525–2025) or configurable year range
- Synthetic-but-accepted directional historical baselines with event overlays
- Three governance modes: NoGov (reactive), AE (Attribution Economics), AE+Bio (AE + biologically-inspired metabolism)
- Split variance metrics: failure variance vs. spend variance
- Spend-parity guardrail for AE/AE+Bio (≈ within +10% of NoGov cumulative spend)
- Region composition (Globe built from continents with weights)
- Reproducible CSV outputs per region/mode + master summary

How to use
----------
from perpetual_filter_sim_v31 import (
    SimulationConfig, SimulationRunner, DEFAULT_REGIONS, DEFAULT_CONTINENTS, DEFAULT_GLOBE_WEIGHTS
)

cfg = SimulationConfig()  # defaults to 1525–2025, presets below
runner = SimulationRunner(cfg)
df_results = runner.run_all(regions=DEFAULT_CONTINENTS + ["Globe"], modes=["NoGov","AE","AE+Bio"])
summary = runner.build_master_summary(df_results)
runner.save_region_bundles(df_results, regions=["Globe"] + DEFAULT_CONTINENTS, tag="v31")

License: CC-BY 4.0 (attribution to the hypothesis author)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import math, json

# -----------------------------
# Configuration dataclasses
# -----------------------------

@dataclass
class Era:
    name: str
    start: int
    end: int
    band: Tuple[float, float]  # (low, high) UE/D range for that era (directional)

@dataclass
class EventOverlay:
    start: int
    end: int
    region: str  # "ALL" or a region/continent name
    add: float   # additive stress on UE/D

@dataclass
class SimulationConfig:
    # Time horizon
    year_start: int = 1525
    year_end: int = 2025

    # Era bands (directional baselines)
    eras: List[Era] = field(default_factory=lambda: [
        Era("E0_PreIndustrial_I", 1525, 1700, (0.20, 0.25)),
        Era("E1_PreIndustrial_II", 1701, 1850, (0.18, 0.22)),
        Era("E2_Industrializing", 1851, 1924, (0.12, 0.18)),
        Era("E3_Compression_I", 1925, 1945, (0.06, 0.11)),
        Era("E4_LongCycle", 1946, 1973, (0.03, 0.04)),
        Era("E5_Compression_II", 1974, 2025, (0.05, 0.10)),
    ])

    # Region multipliers (directional, coarse)
    region_mult: Dict[str, float] = field(default_factory=lambda: {
        "Globe": 1.00,
        "NorthAmerica": 0.95,
        "Europe": 1.05,
        "Asia": 0.90,
        "Africa": 1.20,
        "SouthAmerica": 1.10,
        "Oceania": 0.85,

        # Subregions
        "US": 0.95, "EUCore": 1.00, "EastAsia": 0.85, "SouthAsia": 0.95,
        "SubSaharanAfrica": 1.25, "Andes": 1.10, "SouthernCone": 1.00, "MENA": 1.15,
        "US_California": 0.95, "US_GreatPlains": 0.98, "EU_Nordics": 0.90, "EU_Mediterranean": 1.10,
    })

    # Event overlays
    events: List[EventOverlay] = field(default_factory=lambda: [
        EventOverlay(1816, 1818, "ALL", 0.03),  # Tambora
        EventOverlay(1918, 1919, "ALL", 0.04),  # 1918 flu
        EventOverlay(2020, 2021, "ALL", 0.05),  # COVID

        # Europe
        EventOverlay(1618, 1648, "Europe", 0.05),     # Thirty Years' War
        EventOverlay(1845, 1849, "Europe", 0.03),     # Potato famine & crop failures
        EventOverlay(1914, 1918, "Europe", 0.07),     # WWI
        EventOverlay(1940, 1945, "Europe", 0.10),     # WWII

        # North America
        EventOverlay(1861, 1865, "NorthAmerica", 0.04),  # US Civil War
        EventOverlay(1930, 1933, "NorthAmerica", 0.03),  # Great Depression onset

        # Asia
        EventOverlay(1876, 1879, "Asia", 0.05),       # Great Famine (India/China)
        EventOverlay(1937, 1945, "Asia", 0.04),       # Second Sino-Japanese War

        # Africa
        EventOverlay(1880, 1914, "Africa", 0.02),     # Colonial intensification

        # South America
        EventOverlay(1864, 1870, "SouthAmerica", 0.03),  # War of the Triple Alliance
    ])

    # Globe composition from continents (weights should sum ~1)
    globe_weights: Dict[str, float] = field(default_factory=lambda: {
        "Asia": 0.55, "Africa": 0.17, "Europe": 0.10,
        "NorthAmerica": 0.08, "SouthAmerica": 0.07, "Oceania": 0.03
    })

    # AE guardrails
    spend_parity_cap: float = 0.10  # allow AE/AE+Bio to be within +10% of NoGov cumulative spend

    # AE tuning knobs (neutral defaults; can be tuned per region if desired)
    ae_params: Dict[str, float] = field(default_factory=lambda: {
        "early_warning_tau": 0.85,   # sensitivity to predicted breach (fraction of crisis threshold)
        "crisis_threshold": 0.20,    # notional UE/D at which crisis would trigger
        "precommit_gain": 0.75,      # how strongly AE pre-commit spend reacts to risk
        "effectiveness": 0.65,       # how effective each unit of spend is at lowering UE/D via maintenance
        "wear": 0.012,               # annual wear added to "backlog"
        "latency": 1,                # years ahead AE looks
    })

    # AE+Bio metabolism parameters (multipliers on AE behavior + biology features)
    aebio_params: Dict[str, float] = field(default_factory=lambda: {
        "myelination_gain": 0.90,    # reduces latency over time (cap at 0)
        "immune_budget": 0.02,       # small shock-proximal spend on new anomalies
        "organ_limit_gain": 0.85,    # proportional growth ceiling to prevent sectoral "cancer"
        # three nutrient bands (perishable, semi-durable, durable), contributions sum to ~1
        "nutrients":  {"perish":0.30, "semi":0.45, "durable":0.25},
        "decay":      {"perish":0.25, "semi":0.08, "durable":0.02},  # annual spoilage/obsolescence
        "horizon":    {"perish":2, "semi":7, "durable":30},          # planning horizons (years)
    })

# Defaults
DEFAULT_CONTINENTS = ["NorthAmerica","Europe","Asia","Africa","SouthAmerica","Oceania"]
DEFAULT_REGIONS = DEFAULT_CONTINENTS + ["Globe"]

# -----------------------------
# Historical baseline generator
# -----------------------------

class HistoricalGenerator:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.years = np.arange(cfg.year_start, cfg.year_end+1, dtype=int)

    def generate_region_series(self, region: str) -> pd.DataFrame:
        y = self.years
        base = np.zeros_like(y, dtype=float)
        # era scaffolding
        for era in self.cfg.eras:
            mask = (y >= era.start) & (y <= era.end)
            if mask.sum():
                lo, hi = era.band
                base[mask] = np.linspace(lo, hi, mask.sum())
        # region multiplier
        mult = self.cfg.region_mult.get(region, 1.0)
        base *= mult
        # event overlays
        for ev in self.cfg.events:
            mask = (y >= ev.start) & (y <= ev.end)
            if not mask.sum():
                continue
            if ev.region == "ALL":
                base[mask] += ev.add
            elif ev.region == region:
                base[mask] += ev.add
            else:
                # continental membership heuristic
                cont_map = {
                    "Europe": ["Europe","EUCore","EU_Nordics","EU_Mediterranean"],
                    "NorthAmerica": ["NorthAmerica","US","US_California","US_GreatPlains"],
                    "Asia": ["Asia","EastAsia","SouthAsia","MENA"],
                    "Africa": ["Africa","SubSaharanAfrica","MENA"],
                    "SouthAmerica": ["SouthAmerica","Andes","SouthernCone"],
                    "Oceania": ["Oceania"],
                }
                if region in cont_map.get(ev.region, []):
                    base[mask] += ev.add
        base = np.clip(base, 0.0, 0.75)
        return pd.DataFrame({"region":region, "year":y, "ue_hist":base})

# -----------------------------
# Governance dynamics
# -----------------------------

def simulate_region_mode(region: str,
                         mode: str,
                         hist: pd.DataFrame,
                         cfg: SimulationConfig,
                         tuned_params: Dict[str, float] = None) -> pd.DataFrame:
    """
    Core per-year simulation for a region under a given mode.
    Returns DataFrame with columns: region, mode, year, ueshare, spend
    """
    years = hist["year"].values
    ue_base = hist["ue_hist"].values.copy()

    # governance params
    ae = cfg.ae_params.copy()
    if tuned_params:
        ae.update(tuned_params)

    aebio = cfg.aebio_params.copy()

    # state
    backlog = 0.0       # accumulated maintenance
    cum_spend = 0.0     # cumulative spend for parity checks
    spend_series = []
    ueshare_series = []

    # helpful locals
    def predict_breach(t_idx, latency, ue_current):
        j = min(len(ue_current)-1, t_idx + max(0, int(latency)))
        return ue_current[j]

    # baseline NoGov cumulative spend reference (lazy estimate if not available externally)
    # Using proportional spend to base stress as a placeholder to compare parity.
    # In a full pipeline, we'd pass in the actual NoGov cum spend ref for the region.
    est_nogov_unit_cost = 1.0  # unit cost scaling
    nogov_cum_spend_ref = 0.0

    # Initialization pass to compute a notional NoGov spend path for parity
    if mode in ("AE","AE+Bio"):
        bkl = 0.0
        cs = 0.0
        for t in range(len(years)):
            wear = ae["wear"]
            bkl += wear
            # reactive: only emergency spend if high stress this year
            s = 0.0
            if ue_base[t] + 0.5*bkl > ae["crisis_threshold"]:
                s = 0.04 + 0.06*(ue_base[t])  # emergency spike
                bkl = max(0.0, bkl - 0.4*s)   # reduce a bit
            cs += s * est_nogov_unit_cost
        nogov_cum_spend_ref = cs

    # dynamic variables for AE+Bio
    latency = ae["latency"]
    organ_cap = 1.0

    for t in range(len(years)):
        year = years[t]
        base = ue_base[t]

        # wear accumulation
        backlog += ae["wear"]

        # default: defer maintenance (minimal spend)
        spend = 0.01

        if mode == "NoGov":
            # reactive emergency only
            if base + 0.5*backlog > ae["crisis_threshold"]:
                spend = 0.04 + 0.06*base  # spike during crisis
        else:
            # AE predictive pre-commit
            # predicted stress at (t + latency)
            future = predict_breach(t, latency, ue_base)
            risk = max(0.0, (future / max(1e-6, ae["crisis_threshold"])) - ae["early_warning_tau"])
            precommit = ae["precommit_gain"] * risk

            # baseline AE spend
            spend = 0.01 + precommit

            # AE+Bio metabolism adjustments
            if mode == "AE+Bio":
                # myelination: reduce latency gradually over time
                latency = max(0, latency - aebio["myelination_gain"] * 0.01)
                # immune budget for anomalies (surprise spikes vs trailing mean)
                if t >= 3:
                    recent = ue_base[t-3:t].mean()
                    surprise = max(0.0, base - recent)
                    spend += aebio["immune_budget"] * (surprise / max(1e-6, recent+1e-6))

                # organ proportional limit: if ueshare has been falling and backlog is small,
                # prevent "overinvestment" in one sector analog (coarse emulation)
                # (we model as a soft cap on spend when base is already low)
                organ_cap = aebio["organ_limit_gain"] if base < 0.06 else 1.0
                spend *= organ_cap

            # spend-parity guardrail
            if nogov_cum_spend_ref > 0:
                if cum_spend + spend > (1.0 + cfg.spend_parity_cap) * nogov_cum_spend_ref:
                    # tighten: reduce spend by 25% if breaching parity band
                    spend *= 0.75

        # apply spend to reduce backlog & UE via effectiveness
        backlog = max(0.0, backlog - ae["effectiveness"] * spend)
        ueshare = max(0.0, base + 0.5*backlog - 0.9*ae["effectiveness"] * spend)

        cum_spend += spend
        spend_series.append(spend)
        ueshare_series.append(ueshare)

    out = pd.DataFrame({
        "region": region,
        "mode": mode,
        "year": years,
        "ueshare": np.array(ueshare_series),
        "spend": np.array(spend_series),
    })
    return out

# -----------------------------
# Composition utilities
# -----------------------------

def blend_continents_to_globe(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Rebuild Globe from continent series per mode using weights."""
    out = []
    years = sorted(df["year"].unique())
    for mode in ["NoGov","AE","AE+Bio"]:
        w_ue = np.zeros(len(years))
        w_sp = np.zeros(len(years))
        for cont, w in weights.items():
            sub = df[(df["region"]==cont) & (df["mode"]==mode)]
            if sub.empty:
                continue
            # ensure aligned on years
            sub = sub.sort_values("year")
            yu = sub["ueshare"].values
            ys = sub["spend"].values
            w_ue += w*yu
            w_sp += w*ys
        out.append(pd.DataFrame({
            "region": "Globe",
            "mode": mode,
            "year": years,
            "ueshare": w_ue,
            "spend": w_sp,
        }))
    return pd.concat(out, ignore_index=True)

# -----------------------------
# Scoring / summaries
# -----------------------------

def r2_vs_hist(sim: pd.DataFrame, hist: pd.DataFrame) -> float:
    sim = sim.sort_values("year")
    hist = hist.sort_values("year")
    y = sim["ueshare"].values
    x = hist["ue_hist"].values
    if len(y) != len(x):
        n = min(len(y), len(x))
        y = y[:n]; x = x[:n]
    ss_res = float(np.sum((y-x)**2))
    ss_tot = float(np.sum((x - x.mean())**2) + 1e-12)
    return 1.0 - ss_res/ss_tot

def summarize_region(df_all: pd.DataFrame, hist: pd.DataFrame, region: str) -> Dict[str, Dict[str, float]]:
    out = {}
    for mode in ["NoGov","AE","AE+Bio"]:
        sub = df_all[(df_all["region"]==region) & (df_all["mode"]==mode)].sort_values("year")
        if sub.empty:
            out[mode] = {"peak_ueshare": float("nan"),
                         "variance_failure": float("nan"),
                         "variance_spend": float("nan"),
                         "r2_vs_hist": float("nan"),
                         "cum_spend_vs_nogov": float("nan")}
            continue
        peak = float(sub["ueshare"].max())
        var_fail = float(sub["ueshare"].var())
        var_sp = float(sub["spend"].var())
        r2 = r2_vs_hist(sub, hist[hist["region"]==region])
        # spend parity ratio vs NoGov
        ng = df_all[(df_all["region"]==region) & (df_all["mode"]=="NoGov")]
        ng_cum = float(ng["spend"].sum()) if not ng.empty else 1.0
        cm = float(sub["spend"].sum()) if not sub.empty else ng_cum
        parity = cm / (ng_cum + 1e-12)
        out[mode] = {"peak_ueshare": peak,
                     "variance_failure": var_fail,
                     "variance_spend": var_sp,
                     "r2_vs_hist": r2,
                     "cum_spend_vs_nogov": parity}
    return out

# -----------------------------
# Runner
# -----------------------------

class SimulationRunner:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.histgen = HistoricalGenerator(cfg)

    def run_all(self,
                regions: List[str],
                modes: List[str] = ["NoGov","AE","AE+Bio"],
                tuned_params_by_region: Dict[str, Dict[str, float]] = None) -> pd.DataFrame:
        tuned_params_by_region = tuned_params_by_region or {}
        parts = []
        # Generate historical baselines
        hist_map = {r: self.histgen.generate_region_series(r) for r in regions if r != "Globe"}
        # Simulate continents first
        conts = [r for r in regions if r != "Globe"]
        for r in conts:
            hist = hist_map[r]
            for m in modes:
                params = tuned_params_by_region.get(r, {}).get(m, None)
                sim = simulate_region_mode(r, m, hist, self.cfg, tuned_params=params)
                parts.append(sim)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        # Compose Globe from continents
        if "Globe" in regions:
            globe_df = blend_continents_to_globe(df, self.cfg.globe_weights)
            df = pd.concat([df, globe_df], ignore_index=True)
        return df

    def build_master_summary(self, df_all: pd.DataFrame, regions: List[str] = None) -> pd.DataFrame:
        regions = regions or sorted(df_all["region"].unique().tolist())
        rows = []
        # Build historical refs
        hist_refs = {r: self.histgen.generate_region_series(r) for r in regions if r != "Globe"}
        if "Globe" in regions:
            # Build a pseudo-historical globe baseline by composing the historical baselines with weights
            conts = [r for r in regions if r in self.cfg.globe_weights.keys()]
            yy = sorted(df_all["year"].unique())
            w_ue = np.zeros(len(yy))
            for c in conts:
                h = hist_refs[c]; h = h[h["year"].isin(yy)].sort_values("year")
                w_ue += self.cfg.globe_weights[c] * h["ue_hist"].values
            hist_refs["Globe"] = pd.DataFrame({"region":"Globe","year":yy,"ue_hist":w_ue})

        for r in regions:
            sub_hist = hist_refs[r]
            summ = summarize_region(df_all, pd.concat(hist_refs.values(), ignore_index=True), r)
            for m, vals in summ.items():
                rows.append({"region": r, "mode": m, **vals})
        return pd.DataFrame(rows)

    def save_region_bundles(self, df_all: pd.DataFrame, regions: List[str], tag: str = "v31") -> None:
        # Save per-mode CSVs + summary JSON for each region
        hist_refs = {r: self.histgen.generate_region_series(r) for r in regions if r != "Globe"}
        if "Globe" in regions:
            conts = [r for r in regions if r in self.cfg.globe_weights.keys()]
            yy = sorted(df_all["year"].unique())
            w_ue = np.zeros(len(yy))
            for c in conts:
                h = hist_refs[c]; h = h[h["year"].isin(yy)].sort_values("year")
                w_ue += self.cfg.globe_weights[c] * h["ue_hist"].values
            hist_refs["Globe"] = pd.DataFrame({"region":"Globe","year":yy,"ue_hist":w_ue})

        for r in regions:
            sub = df_all[df_all["region"]==r].copy()
            for m in ["NoGov","AE","AE+Bio"]:
                sub[sub["mode"]==m].to_csv(f"/mnt/data/{tag}_{r.lower()}_{m.lower()}.csv", index=False)
            summ = summarize_region(df_all, pd.concat(hist_refs.values(), ignore_index=True), r)
            with open(f"/mnt/data/{tag}_{r.lower()}_summary.json","w") as f:
                json.dump(summ, f, indent=2)

# End of module
