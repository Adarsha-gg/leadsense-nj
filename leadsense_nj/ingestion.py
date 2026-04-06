from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


CENSUS_ACS_BASE_URL = "https://api.census.gov/data"
EPA_EFSERVICE_BASE_URL = "https://data.epa.gov/efservice"


ACS_GET_COLUMNS: tuple[str, ...] = (
    "B19013_001E",  # median household income
    "B25034_001E",  # total housing units by year built
    "B25034_010E",  # 1940-1949
    "B25034_011E",  # 1939 or earlier
    "B25035_001E",  # median year structure built
    "B17001_001E",  # poverty denominator
    "B17001_002E",  # poverty numerator
    "B01001_001E",  # total population
    "B01001_003E",  # male under 5
    "B01001_027E",  # female under 5
    "B02001_001E",  # race total
    "B02001_002E",  # white alone
)


@dataclass(frozen=True)
class IngestionArtifacts:
    acs_path: Path
    epa_lcr_samples_path: Path
    epa_lcr_sample_results_path: Path
    epa_violations_path: Path
    epa_pws_summary_path: Path
    metadata_path: Path


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    den = den.replace(0.0, np.nan)
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.clip(lower=0.0, upper=1.0)


def _sanitize_acs_numeric(df: pd.DataFrame, cols: tuple[str, ...] = ACS_GET_COLUMNS) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        # ACS uses large negative sentinels for missing values.
        out[col] = out[col].where(out[col] >= 0, np.nan)
    return out


def validate_acs_block_group_frame(df: pd.DataFrame) -> None:
    required = [
        "geoid",
        "median_income",
        "pct_housing_pre_1950",
        "poverty_rate",
        "children_under_6_rate",
        "minority_share",
        "median_housing_year",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"ACS frame missing required columns: {missing}")

    geoid = df["geoid"].astype(str)
    if not geoid.str.fullmatch(r"\d{12}").all():
        raise ValueError("ACS frame contains invalid GEOID values.")
    if geoid.duplicated().any():
        raise ValueError("ACS frame contains duplicate GEOID values.")

    for rate_col in ["pct_housing_pre_1950", "poverty_rate", "children_under_6_rate", "minority_share"]:
        series = pd.to_numeric(df[rate_col], errors="coerce")
        bad = ~series.isna() & ((series < 0.0) | (series > 1.0))
        if bool(bad.any()):
            raise ValueError(f"ACS frame has out-of-range values in {rate_col}.")


def fetch_census_acs_block_groups(
    *,
    year: int = 2022,
    state_fips: str = "34",
    api_key: str | None = None,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    sess = session or requests.Session()
    key = api_key or os.getenv("CENSUS_API_KEY")
    params: list[tuple[str, str]] = [
        ("get", ",".join(ACS_GET_COLUMNS)),
        ("for", "block group:*"),
        ("in", f"state:{state_fips}"),
        ("in", "county:*"),
        ("in", "tract:*"),
    ]
    if key:
        params.append(("key", key))

    url = f"{CENSUS_ACS_BASE_URL}/{year}/acs/acs5"
    response = sess.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("Unexpected ACS API payload format.")

    header = payload[0]
    rows = payload[1:]
    raw = pd.DataFrame(rows, columns=header)
    raw = _sanitize_acs_numeric(raw)
    raw["geoid"] = raw["state"].astype(str) + raw["county"].astype(str) + raw["tract"].astype(str) + raw["block group"].astype(str)

    out = pd.DataFrame(
        {
            "geoid": raw["geoid"].astype(str),
            "median_income": raw["B19013_001E"],
            "pct_housing_pre_1950": _safe_rate(raw["B25034_010E"] + raw["B25034_011E"], raw["B25034_001E"]),
            "poverty_rate": _safe_rate(raw["B17001_002E"], raw["B17001_001E"]),
            # ACS B01001 has under-5 bins; used here as an under-6 proxy.
            "children_under_6_rate": _safe_rate(raw["B01001_003E"] + raw["B01001_027E"], raw["B01001_001E"]),
            "minority_share": (1.0 - _safe_rate(raw["B02001_002E"], raw["B02001_001E"])).clip(lower=0.0, upper=1.0),
            "median_housing_year": raw["B25035_001E"],
            "acs_year": int(year),
        }
    ).sort_values("geoid")

    validate_acs_block_group_frame(out)
    return out.reset_index(drop=True)


def _build_efservice_url(
    table: str,
    *,
    filters: list[tuple[str, str]] | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> str:
    parts = [EPA_EFSERVICE_BASE_URL, table]
    for key, value in filters or []:
        parts.extend([key, value])
    if row_start is not None and row_end is not None:
        parts.extend(["rows", f"{row_start}:{row_end}"])
    parts.append("CSV")
    return "/".join(parts)


def fetch_epa_efservice_table(
    table: str,
    *,
    filters: list[tuple[str, str]] | None = None,
    row_start: int = 0,
    row_end: int = 10000,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    sess = session or requests.Session()
    url = _build_efservice_url(table, filters=filters, row_start=row_start, row_end=row_end)
    response = sess.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    text = response.text.strip()
    if text.startswith("{") and '"error"' in text:
        raise RuntimeError(f"EPA efservice error for table '{table}': {text}")
    if not text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def fetch_epa_efservice_table_paged(
    table: str,
    *,
    filters: list[tuple[str, str]] | None = None,
    max_rows: int = 30000,
    chunk_size: int = 10000,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    if max_rows <= 0:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    start = 0
    while start < max_rows:
        end = min(start + chunk_size, max_rows)  # inclusive window handled below
        chunk = fetch_epa_efservice_table(
            table,
            filters=filters,
            row_start=start,
            row_end=end,
            timeout_seconds=timeout_seconds,
            session=session,
        )
        if chunk.empty:
            break

        chunks.append(chunk)
        requested_rows = end - start + 1
        if len(chunk) < requested_rows:
            break
        start = end + 1

    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out.iloc[:max_rows].reset_index(drop=True)


def _to_lead_ppb(series: pd.Series, unit_series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    units = unit_series.fillna("").astype(str).str.lower()
    mg_l_mask = units.eq("mg/l")
    values.loc[mg_l_mask] = values.loc[mg_l_mask] * 1000.0
    return values


def build_epa_pws_lead_signals(
    lcr_samples: pd.DataFrame,
    lcr_sample_results: pd.DataFrame,
    *,
    violations: pd.DataFrame | None = None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    sample_required = {"pwsid", "sample_id", "sampling_end_date"}
    result_required = {"pwsid", "sample_id", "contaminant_code", "sample_measure", "unit_of_measure"}
    if not sample_required.issubset(set(lcr_samples.columns)):
        missing = sorted(sample_required.difference(lcr_samples.columns))
        raise ValueError(f"LCR sample frame missing required columns: {missing}")
    if not result_required.issubset(set(lcr_sample_results.columns)):
        missing = sorted(result_required.difference(lcr_sample_results.columns))
        raise ValueError(f"LCR sample result frame missing required columns: {missing}")

    now = reference_date or pd.Timestamp.now(tz="UTC").tz_localize(None)
    cutoff_5y = now - pd.DateOffset(years=5)
    cutoff_3y = now - pd.DateOffset(years=3)

    samples = lcr_samples.copy()
    samples["sampling_end_date"] = pd.to_datetime(samples["sampling_end_date"], errors="coerce")
    samples = samples.dropna(subset=["pwsid", "sample_id"]).copy()

    results = lcr_sample_results.copy()
    results = results.dropna(subset=["pwsid", "sample_id"]).copy()
    results["contaminant_code"] = results["contaminant_code"].astype(str).str.upper()
    lead_results = results[results["contaminant_code"].str.startswith("PB")].copy()
    lead_results["lead_ppb"] = _to_lead_ppb(lead_results["sample_measure"], lead_results["unit_of_measure"])

    merged = lead_results.merge(
        samples[["pwsid", "sample_id", "sampling_end_date"]],
        on=["pwsid", "sample_id"],
        how="left",
    )

    merged = merged.dropna(subset=["pwsid", "lead_ppb"]).copy()
    merged["sampling_end_date"] = pd.to_datetime(merged["sampling_end_date"], errors="coerce")

    in_5y = merged["sampling_end_date"] >= cutoff_5y
    in_3y = merged["sampling_end_date"] >= cutoff_3y

    grouped = merged.groupby("pwsid", dropna=True)
    summary = pd.DataFrame({"pwsid": sorted(merged["pwsid"].astype(str).unique())})
    summary = summary.set_index("pwsid")
    summary["pws_action_level_exceedance_5y"] = grouped.apply(
        lambda g: bool(((g["lead_ppb"] > 15.0) & (g["sampling_end_date"] >= cutoff_5y)).any())
    )
    summary["pws_any_sample_gt15_3y"] = grouped.apply(
        lambda g: bool(((g["lead_ppb"] > 15.0) & (g["sampling_end_date"] >= cutoff_3y)).any())
    )
    summary["pws_lead_90p_latest_ppb"] = grouped.apply(
        lambda g: float(g.sort_values("sampling_end_date")["lead_ppb"].iloc[-1]) if len(g) > 0 else np.nan
    )
    summary["pws_sample_count_5y"] = grouped.apply(
        lambda g: int((g["sampling_end_date"] >= cutoff_5y).sum())
    )

    if violations is not None and not violations.empty and {"pwsid", "compl_per_end_date"}.issubset(violations.columns):
        viol = violations.copy()
        viol["compl_per_end_date"] = pd.to_datetime(viol["compl_per_end_date"], errors="coerce")
        if "contaminant_code" in viol.columns:
            contaminant = pd.to_numeric(viol["contaminant_code"], errors="coerce")
            viol = viol[contaminant.eq(1030)]
        viol_5y = viol[viol["compl_per_end_date"] >= cutoff_5y]
        viol_counts = viol_5y.groupby("pwsid").size().rename("pws_violation_count_5y")
        summary = summary.join(viol_counts, how="left")
        summary["pws_violation_count_5y"] = summary["pws_violation_count_5y"].fillna(0).astype(int)
    else:
        summary["pws_violation_count_5y"] = 0

    out = summary.reset_index()
    out["pws_action_level_exceedance_5y"] = out["pws_action_level_exceedance_5y"].fillna(False).astype(bool)
    out["pws_any_sample_gt15_3y"] = out["pws_any_sample_gt15_3y"].fillna(False).astype(bool)
    return out.sort_values("pwsid").reset_index(drop=True)


def validate_epa_pws_lead_signal_frame(df: pd.DataFrame) -> None:
    required = [
        "pwsid",
        "pws_action_level_exceedance_5y",
        "pws_any_sample_gt15_3y",
        "pws_lead_90p_latest_ppb",
        "pws_sample_count_5y",
        "pws_violation_count_5y",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"EPA PWS lead signal frame missing required columns: {missing}")
    if df.empty:
        raise ValueError("EPA PWS lead signal frame is empty.")
    if df["pwsid"].astype(str).str.len().eq(0).any():
        raise ValueError("EPA PWS lead signal frame contains blank pwsid values.")


def build_real_data_cache(
    *,
    cache_dir: Path | str = Path("data") / "cache",
    acs_year: int = 2022,
    state_fips: str = "34",
    max_violation_rows: int = 30000,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> IngestionArtifacts:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    acs_df = fetch_census_acs_block_groups(
        year=acs_year,
        state_fips=state_fips,
        timeout_seconds=timeout_seconds,
        session=session,
    )
    lcr_samples_df = fetch_epa_efservice_table_paged(
        "LCR_SAMPLE",
        filters=[("PRIMACY_AGENCY_CODE", "NJ")],
        max_rows=50000,
        timeout_seconds=timeout_seconds,
        session=session,
    )
    lcr_sample_results_df = fetch_epa_efservice_table_paged(
        "LCR_SAMPLE_RESULT",
        filters=[("PRIMACY_AGENCY_CODE", "NJ")],
        max_rows=50000,
        timeout_seconds=timeout_seconds,
        session=session,
    )
    violations_df = fetch_epa_efservice_table_paged(
        "VIOLATION",
        filters=[("PRIMACY_AGENCY_CODE", "NJ"), ("CONTAMINANT_CODE", "1030")],
        max_rows=max_violation_rows,
        timeout_seconds=timeout_seconds,
        session=session,
    )

    pws_summary_df = build_epa_pws_lead_signals(
        lcr_samples=lcr_samples_df,
        lcr_sample_results=lcr_sample_results_df,
        violations=violations_df,
    )
    validate_epa_pws_lead_signal_frame(pws_summary_df)

    acs_out = cache_path / f"acs_nj_block_groups_{acs_year}.csv"
    lcr_samples_out = cache_path / "epa_nj_lcr_samples.csv"
    lcr_results_out = cache_path / "epa_nj_lcr_sample_results.csv"
    violations_out = cache_path / "epa_nj_lead_violations.csv"
    pws_summary_out = cache_path / "epa_nj_pws_lead_signals.csv"
    metadata_out = cache_path / "ingestion_metadata.json"

    acs_df.to_csv(acs_out, index=False)
    lcr_samples_df.to_csv(lcr_samples_out, index=False)
    lcr_sample_results_df.to_csv(lcr_results_out, index=False)
    violations_df.to_csv(violations_out, index=False)
    pws_summary_df.to_csv(pws_summary_out, index=False)

    metadata: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "acs_year": acs_year,
        "state_fips": state_fips,
        "sources": {
            "census_acs": f"{CENSUS_ACS_BASE_URL}/{acs_year}/acs/acs5",
            "epa_lcr_sample": _build_efservice_url("LCR_SAMPLE", filters=[("PRIMACY_AGENCY_CODE", "NJ")]),
            "epa_lcr_sample_result": _build_efservice_url("LCR_SAMPLE_RESULT", filters=[("PRIMACY_AGENCY_CODE", "NJ")]),
            "epa_violation_lead": _build_efservice_url(
                "VIOLATION", filters=[("PRIMACY_AGENCY_CODE", "NJ"), ("CONTAMINANT_CODE", "1030")]
            ),
        },
        "row_counts": {
            "acs_block_groups": int(len(acs_df)),
            "epa_lcr_samples": int(len(lcr_samples_df)),
            "epa_lcr_sample_results": int(len(lcr_sample_results_df)),
            "epa_lead_violations": int(len(violations_df)),
            "epa_pws_summary": int(len(pws_summary_df)),
        },
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return IngestionArtifacts(
        acs_path=acs_out,
        epa_lcr_samples_path=lcr_samples_out,
        epa_lcr_sample_results_path=lcr_results_out,
        epa_violations_path=violations_out,
        epa_pws_summary_path=pws_summary_out,
        metadata_path=metadata_out,
    )


def ensure_real_data_cache(
    *,
    cache_dir: Path | str = Path("data") / "cache",
    acs_year: int = 2022,
    refresh: bool = False,
    max_violation_rows: int = 30000,
    timeout_seconds: int = 60,
) -> IngestionArtifacts:
    cache_path = Path(cache_dir)
    artifacts = IngestionArtifacts(
        acs_path=cache_path / f"acs_nj_block_groups_{acs_year}.csv",
        epa_lcr_samples_path=cache_path / "epa_nj_lcr_samples.csv",
        epa_lcr_sample_results_path=cache_path / "epa_nj_lcr_sample_results.csv",
        epa_violations_path=cache_path / "epa_nj_lead_violations.csv",
        epa_pws_summary_path=cache_path / "epa_nj_pws_lead_signals.csv",
        metadata_path=cache_path / "ingestion_metadata.json",
    )
    files_exist = all(path.exists() for path in artifacts.__dict__.values())
    if refresh or not files_exist:
        return build_real_data_cache(
            cache_dir=cache_path,
            acs_year=acs_year,
            max_violation_rows=max_violation_rows,
            timeout_seconds=timeout_seconds,
        )
    return artifacts
