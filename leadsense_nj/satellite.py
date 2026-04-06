from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


PLANETARY_COMPUTER_STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


@dataclass(frozen=True)
class SentinelCacheArtifacts:
    features_path: Path
    metadata_path: Path


def build_bbox_from_point(lat: float, lon: float, *, half_size_deg: float = 0.01) -> list[float]:
    lat_f = float(lat)
    lon_f = float(lon)
    delta = float(max(half_size_deg, 1e-5))
    return [lon_f - delta, lat_f - delta, lon_f + delta, lat_f + delta]


def search_sentinel2_items(
    *,
    bbox: list[float],
    start_date: str,
    end_date: str,
    limit: int = 3,
    max_cloud_cover: float = 50.0,
    collection: str = "sentinel-2-l2a",
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    if len(bbox) != 4:
        raise ValueError("bbox must have four coordinates: [min_lon, min_lat, max_lon, max_lat]")
    sess = session or requests.Session()
    payload = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start_date}/{end_date}",
        "limit": int(max(1, limit)),
        "query": {"eo:cloud_cover": {"lt": float(max_cloud_cover)}},
    }
    response = sess.post(PLANETARY_COMPUTER_STAC_SEARCH_URL, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    body = response.json()
    return list(body.get("features", []))


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float("nan")


def extract_sentinel_item_features(item: dict[str, Any]) -> dict[str, Any]:
    props = item.get("properties", {}) or {}
    dt = pd.to_datetime(props.get("datetime"), errors="coerce")
    return {
        "item_id": str(item.get("id", "")),
        "item_datetime": dt,
        "cloud_cover": _safe_float(props.get("eo:cloud_cover")),
        "vegetation_pct": _safe_float(props.get("s2:vegetation_percentage")),
        "water_pct": _safe_float(props.get("s2:water_percentage")),
        "nodata_pct": _safe_float(props.get("s2:nodata_pixel_percentage")),
        "sun_elevation": _safe_float(props.get("view:sun_elevation")),
        "sun_azimuth": _safe_float(props.get("view:sun_azimuth")),
    }


def aggregate_sentinel_tile_features(
    items: list[dict[str, Any]],
    *,
    reference_date: pd.Timestamp | None = None,
) -> dict[str, float]:
    if not items:
        return {
            "s2_item_count": 0.0,
            "s2_cloud_cover_mean": 100.0,
            "s2_vegetation_pct_mean": 0.0,
            "s2_water_pct_mean": 0.0,
            "s2_nodata_pct_mean": 100.0,
            "s2_sun_elevation_mean": 0.0,
            "s2_days_since_latest": 3650.0,
        }

    rows = pd.DataFrame([extract_sentinel_item_features(item) for item in items])
    numeric_cols = [
        "cloud_cover",
        "vegetation_pct",
        "water_pct",
        "nodata_pct",
        "sun_elevation",
    ]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")

    if reference_date is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    else:
        now = pd.to_datetime(reference_date, utc=True, errors="coerce")
        now = now.tz_convert(None) if getattr(now, "tzinfo", None) is not None else now
    datetimes = pd.to_datetime(rows["item_datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    latest = datetimes.max()
    if pd.isna(latest):
        days_since = 3650.0
    else:
        days_since = max(float((now - latest).days), 0.0)

    return {
        "s2_item_count": float(len(rows)),
        "s2_cloud_cover_mean": float(rows["cloud_cover"].fillna(rows["cloud_cover"].median(skipna=True)).mean()),
        "s2_vegetation_pct_mean": float(
            rows["vegetation_pct"].fillna(rows["vegetation_pct"].median(skipna=True)).mean()
        ),
        "s2_water_pct_mean": float(rows["water_pct"].fillna(rows["water_pct"].median(skipna=True)).mean()),
        "s2_nodata_pct_mean": float(rows["nodata_pct"].fillna(rows["nodata_pct"].median(skipna=True)).mean()),
        "s2_sun_elevation_mean": float(
            rows["sun_elevation"].fillna(rows["sun_elevation"].median(skipna=True)).mean()
        ),
        "s2_days_since_latest": days_since,
    }


def fetch_sentinel_features_for_block_groups(
    df: pd.DataFrame,
    *,
    geoid_col: str = "geoid",
    lat_col: str = "lat",
    lon_col: str = "lon",
    start_date: str = "2024-04-01",
    end_date: str = "2024-10-31",
    items_per_block: int = 2,
    max_cloud_cover: float = 50.0,
    bbox_half_size_deg: float = 0.01,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
    allow_fallback: bool = True,
) -> pd.DataFrame:
    required = {geoid_col, lat_col, lon_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for Sentinel feature fetch: {missing}")

    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        geoid = str(row[geoid_col])
        lat = _safe_float(row[lat_col])
        lon = _safe_float(row[lon_col])
        if not np.isfinite(lat) or not np.isfinite(lon):
            agg = aggregate_sentinel_tile_features([])
            agg["s2_source"] = "fallback_missing_coords"
            out_rows.append({"geoid": geoid, **agg})
            continue

        bbox = build_bbox_from_point(lat, lon, half_size_deg=bbox_half_size_deg)
        try:
            items = search_sentinel2_items(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                limit=items_per_block,
                max_cloud_cover=max_cloud_cover,
                timeout_seconds=timeout_seconds,
                session=session,
            )
            if not items and allow_fallback:
                agg = aggregate_sentinel_tile_features([])
                source = "fallback_no_items"
            else:
                agg = aggregate_sentinel_tile_features(items)
                source = "live"
        except Exception:
            if not allow_fallback:
                raise
            agg = aggregate_sentinel_tile_features([])
            source = "fallback_error"

        agg["s2_source"] = source
        out_rows.append({"geoid": geoid, **agg})

    out = pd.DataFrame(out_rows)
    metric_cols = [col for col in out.columns if col.startswith("s2_") and col != "s2_source"]
    for col in metric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out.sort_values("geoid").reset_index(drop=True)


def validate_sentinel_feature_frame(df: pd.DataFrame) -> None:
    required = [
        "geoid",
        "s2_item_count",
        "s2_cloud_cover_mean",
        "s2_vegetation_pct_mean",
        "s2_water_pct_mean",
        "s2_nodata_pct_mean",
        "s2_sun_elevation_mean",
        "s2_days_since_latest",
        "s2_source",
    ]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Sentinel feature frame missing required columns: {missing}")
    if df.empty:
        raise ValueError("Sentinel feature frame is empty.")
    if not df["geoid"].astype(str).str.fullmatch(r"\d{1,12}").all():
        raise ValueError("Sentinel feature frame has invalid geoid values.")
    if (pd.to_numeric(df["s2_item_count"], errors="coerce") < 0).any():
        raise ValueError("Sentinel feature frame has negative s2_item_count.")


def build_sentinel_feature_cache(
    df: pd.DataFrame,
    *,
    cache_dir: Path | str = Path("data") / "cache",
    features_filename: str = "sentinel_features_sample.csv",
    metadata_filename: str = "sentinel_features_metadata.json",
    start_date: str = "2024-04-01",
    end_date: str = "2024-10-31",
    items_per_block: int = 2,
    max_cloud_cover: float = 50.0,
    bbox_half_size_deg: float = 0.01,
    timeout_seconds: int = 60,
    session: requests.Session | None = None,
    allow_fallback: bool = True,
) -> SentinelCacheArtifacts:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    features_path = cache_path / features_filename
    metadata_path = cache_path / metadata_filename

    features_df = fetch_sentinel_features_for_block_groups(
        df,
        start_date=start_date,
        end_date=end_date,
        items_per_block=items_per_block,
        max_cloud_cover=max_cloud_cover,
        bbox_half_size_deg=bbox_half_size_deg,
        timeout_seconds=timeout_seconds,
        session=session,
        allow_fallback=allow_fallback,
    )
    validate_sentinel_feature_frame(features_df)
    features_df.to_csv(features_path, index=False)

    metadata = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stac_search_url": PLANETARY_COMPUTER_STAC_SEARCH_URL,
        "start_date": start_date,
        "end_date": end_date,
        "items_per_block": int(items_per_block),
        "max_cloud_cover": float(max_cloud_cover),
        "bbox_half_size_deg": float(bbox_half_size_deg),
        "rows": int(len(features_df)),
        "sources": features_df["s2_source"].value_counts().to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SentinelCacheArtifacts(features_path=features_path, metadata_path=metadata_path)


def ensure_sentinel_feature_cache(
    df: pd.DataFrame,
    *,
    cache_dir: Path | str = Path("data") / "cache",
    features_filename: str = "sentinel_features_sample.csv",
    metadata_filename: str = "sentinel_features_metadata.json",
    refresh: bool = False,
    start_date: str = "2024-04-01",
    end_date: str = "2024-10-31",
    items_per_block: int = 2,
    max_cloud_cover: float = 50.0,
    bbox_half_size_deg: float = 0.01,
    timeout_seconds: int = 60,
    allow_fallback: bool = True,
) -> SentinelCacheArtifacts:
    cache_path = Path(cache_dir)
    features_path = cache_path / features_filename
    metadata_path = cache_path / metadata_filename
    if not refresh and features_path.exists() and metadata_path.exists():
        loaded = pd.read_csv(features_path, dtype={"geoid": str})
        validate_sentinel_feature_frame(loaded)
        return SentinelCacheArtifacts(features_path=features_path, metadata_path=metadata_path)

    return build_sentinel_feature_cache(
        df,
        cache_dir=cache_path,
        features_filename=features_filename,
        metadata_filename=metadata_filename,
        start_date=start_date,
        end_date=end_date,
        items_per_block=items_per_block,
        max_cloud_cover=max_cloud_cover,
        bbox_half_size_deg=bbox_half_size_deg,
        timeout_seconds=timeout_seconds,
        allow_fallback=allow_fallback,
    )
