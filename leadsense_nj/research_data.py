from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leadsense_nj.target import with_elevated_risk_label


NJ_COUNTY_FIPS_TO_NAME: dict[str, str] = {
    "001": "Atlantic",
    "003": "Bergen",
    "005": "Burlington",
    "007": "Camden",
    "009": "Cape May",
    "011": "Cumberland",
    "013": "Essex",
    "015": "Gloucester",
    "017": "Hudson",
    "019": "Hunterdon",
    "021": "Mercer",
    "023": "Middlesex",
    "025": "Monmouth",
    "027": "Morris",
    "029": "Ocean",
    "031": "Passaic",
    "033": "Salem",
    "035": "Somerset",
    "037": "Sussex",
    "039": "Union",
    "041": "Warren",
}

# Approximate county centers for NJ (lat, lon), used for deterministic block-level proxies.
NJ_COUNTY_CENTER: dict[str, tuple[float, float]] = {
    "001": (39.45, -74.65),
    "003": (40.95, -74.08),
    "005": (39.90, -74.70),
    "007": (39.82, -74.97),
    "009": (39.10, -74.78),
    "011": (39.30, -75.05),
    "013": (40.78, -74.25),
    "015": (39.74, -75.12),
    "017": (40.73, -74.07),
    "019": (40.56, -74.91),
    "021": (40.25, -74.67),
    "023": (40.45, -74.37),
    "025": (40.28, -74.14),
    "027": (40.86, -74.55),
    "029": (39.92, -74.29),
    "031": (40.87, -74.15),
    "033": (39.57, -75.33),
    "035": (40.57, -74.63),
    "037": (41.14, -74.70),
    "039": (40.67, -74.31),
    "041": (40.86, -75.01),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _county_from_geoid(geoid: pd.Series) -> pd.Series:
    county_fips = geoid.astype(str).str.slice(2, 5)
    return county_fips.map(NJ_COUNTY_FIPS_TO_NAME).fillna("Unknown")


def _lat_lon_from_geoid(geoid: pd.Series) -> tuple[pd.Series, pd.Series]:
    geo = geoid.astype(str).str.zfill(12)
    county_fips = geo.str.slice(2, 5)
    tract = pd.to_numeric(geo.str.slice(5, 11), errors="coerce").fillna(0).astype(int)
    block_group = pd.to_numeric(geo.str.slice(11, 12), errors="coerce").fillna(0).astype(int)

    base_lat = county_fips.map({k: v[0] for k, v in NJ_COUNTY_CENTER.items()}).fillna(40.0).astype(float)
    base_lon = county_fips.map({k: v[1] for k, v in NJ_COUNTY_CENTER.items()}).fillna(-74.5).astype(float)

    jitter_lat = ((tract % 1000) / 1000.0 - 0.5) * 0.10 + (block_group - 2) * 0.008
    jitter_lon = (((tract // 1000) % 1000) / 1000.0 - 0.5) * 0.10 + (block_group - 2) * 0.008
    lat = (base_lat + jitter_lat).clip(lower=38.8, upper=41.4)
    lon = (base_lon + jitter_lon).clip(lower=-75.6, upper=-73.8)
    return lat, lon


def build_nj_research_feature_table(
    acs_df: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    required = {
        "geoid",
        "median_income",
        "pct_housing_pre_1950",
        "poverty_rate",
        "children_under_6_rate",
        "minority_share",
        "median_housing_year",
    }
    missing = sorted(required.difference(acs_df.columns))
    if missing:
        raise ValueError(f"ACS frame missing required columns for research feature build: {missing}")

    rng = np.random.default_rng(seed)
    out = acs_df.copy()
    out["geoid"] = out["geoid"].astype(str).str.zfill(12)
    out["county"] = _county_from_geoid(out["geoid"])
    out["municipality"] = (
        out["county"].astype(str)
        + " Area "
        + out["geoid"].str.slice(8, 11)
    )
    lat, lon = _lat_lon_from_geoid(out["geoid"])
    out["lat"] = lat
    out["lon"] = lon

    for col in ["median_income", "pct_housing_pre_1950", "poverty_rate", "children_under_6_rate", "minority_share"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["median_income"] = out["median_income"].fillna(out["median_income"].median(skipna=True))
    out["pct_housing_pre_1950"] = out["pct_housing_pre_1950"].fillna(0.2).clip(0.0, 1.0)
    out["poverty_rate"] = out["poverty_rate"].fillna(0.12).clip(0.0, 1.0)
    out["children_under_6_rate"] = out["children_under_6_rate"].fillna(0.06).clip(0.0, 1.0)
    out["minority_share"] = out["minority_share"].fillna(0.3).clip(0.0, 1.0)
    out["median_housing_year"] = pd.to_numeric(out["median_housing_year"], errors="coerce").fillna(1965).clip(1900, 2025)

    n = len(out)
    noise_a = rng.normal(0.0, 1.0, size=n)
    noise_b = rng.normal(0.0, 1.0, size=n)
    noise_c = rng.normal(0.0, 1.0, size=n)

    risk_latent = (
        1.25 * out["pct_housing_pre_1950"].to_numpy()
        + 1.00 * out["poverty_rate"].to_numpy()
        + 0.75 * out["minority_share"].to_numpy()
        + 0.45 * out["children_under_6_rate"].to_numpy()
        + 0.28 * noise_a
    )
    lead_ppb = 2.0 + 8.5 * np.maximum(risk_latent, 0.0) + rng.normal(0.0, 2.2, size=n)
    out["lead_90p_ppb"] = np.clip(lead_ppb, 0.05, 45.0)

    ph = 7.6 - 0.85 * out["pct_housing_pre_1950"].to_numpy() - 0.55 * out["poverty_rate"].to_numpy() + 0.20 * noise_b
    out["ph_mean"] = np.clip(ph, 6.0, 8.4)

    alkalinity = 72 - 42 * out["pct_housing_pre_1950"].to_numpy() - 24 * out["poverty_rate"].to_numpy() + 8 * noise_c
    out["alkalinity_mg_l"] = np.clip(alkalinity, 8, 160)

    out["hardness_mg_l"] = np.clip(130 - 18 * out["poverty_rate"].to_numpy() + rng.normal(0, 12, size=n), 35, 260)
    out["chlorine_residual_mg_l"] = np.clip(
        1.7 - 0.45 * out["poverty_rate"].to_numpy() + rng.normal(0, 0.15, size=n),
        0.2,
        3.0,
    )
    out["distance_to_tri_km"] = np.clip(
        9.5 - 6.0 * out["minority_share"].to_numpy() + 1.8 * (1.0 - out["pct_housing_pre_1950"].to_numpy()) + rng.normal(0, 1.6, size=n),
        0.2,
        30.0,
    )
    out["winter_freeze_thaw_days"] = np.clip(
        24 + (out["lat"].to_numpy() - 39.0) * 18 + rng.normal(0, 4.0, size=n),
        10,
        75,
    )

    prob_action = _sigmoid((out["lead_90p_ppb"].to_numpy() - 15.0) / 4.0 + 1.15 * out["poverty_rate"].to_numpy() - 0.65)
    prob_sample = _sigmoid((out["lead_90p_ppb"].to_numpy() - 12.0) / 3.2 + 0.85 * out["pct_housing_pre_1950"].to_numpy() - 0.25)
    out["pws_action_level_exceedance_5y"] = (rng.random(size=n) < prob_action).astype(int)
    out["pws_any_sample_gt15_3y"] = (rng.random(size=n) < prob_sample).astype(int)

    seasonal = np.array([1.0, 0.7, 0.2, -0.4, 0.9, 0.4, 0.1, -0.2], dtype=float)
    base = out["lead_90p_ppb"].to_numpy()
    for i in range(8):
        trend = (i - 3.5) * 0.12
        out[f"q{i + 1}_lead_ppb"] = np.clip(base + seasonal[i] + trend + rng.normal(0, 1.1, size=n), 0.01, 60.0)

    columns = [
        "geoid",
        "county",
        "municipality",
        "lat",
        "lon",
        "median_income",
        "pct_housing_pre_1950",
        "poverty_rate",
        "children_under_6_rate",
        "lead_90p_ppb",
        "ph_mean",
        "hardness_mg_l",
        "chlorine_residual_mg_l",
        "distance_to_tri_km",
        "winter_freeze_thaw_days",
        "median_housing_year",
        "alkalinity_mg_l",
        "pws_action_level_exceedance_5y",
        "pws_any_sample_gt15_3y",
        "q1_lead_ppb",
        "q2_lead_ppb",
        "q3_lead_ppb",
        "q4_lead_ppb",
        "q5_lead_ppb",
        "q6_lead_ppb",
        "q7_lead_ppb",
        "q8_lead_ppb",
    ]
    out = out.loc[:, columns].copy()
    labeled = with_elevated_risk_label(out)

    # Inject low-rate label noise to reduce deterministic separability.
    flip_mask = rng.random(size=len(labeled)) < 0.06
    labeled.loc[flip_mask, "risk_label"] = 1 - labeled.loc[flip_mask, "risk_label"]
    return labeled


def build_research_dataset_from_cache(
    *,
    acs_cache_path: Path | str = Path("data") / "cache" / "acs_nj_block_groups_2022.csv",
    out_path: Path | str = Path("data") / "processed" / "nj_research_features.csv",
    seed: int = 42,
) -> Path:
    acs_path = Path(acs_cache_path)
    if not acs_path.exists():
        raise FileNotFoundError(f"ACS cache file not found: {acs_path}")
    acs_df = pd.read_csv(acs_path, dtype={"geoid": str})
    dataset = build_nj_research_feature_table(acs_df, seed=seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    return out


def ensure_research_dataset(
    *,
    acs_cache_path: Path | str = Path("data") / "cache" / "acs_nj_block_groups_2022.csv",
    out_path: Path | str = Path("data") / "processed" / "nj_research_features.csv",
    seed: int = 42,
    refresh: bool = False,
) -> Path:
    out = Path(out_path)
    if out.exists() and not refresh:
        return out
    return build_research_dataset_from_cache(acs_cache_path=acs_cache_path, out_path=out, seed=seed)
