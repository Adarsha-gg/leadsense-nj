from __future__ import annotations

import pandas as pd


def normalize_gps_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    required = {"timestamp", "lat", "lon"}
    if not required.issubset(set(cols.keys())):
        raise ValueError("GPS CSV must contain columns: timestamp, lat, lon")
    out = df[[cols["timestamp"], cols["lat"], cols["lon"]]].copy()
    out.columns = ["timestamp", "lat", "lon"]
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna().sort_values("timestamp")
    return out.reset_index(drop=True)


def interpolate_lat_lon(gps_df: pd.DataFrame, timestamp_s: float) -> tuple[float | None, float | None]:
    if gps_df.empty:
        return None, None
    left = gps_df[gps_df["timestamp"] <= timestamp_s].tail(1)
    right = gps_df[gps_df["timestamp"] >= timestamp_s].head(1)
    if left.empty and right.empty:
        return None, None
    if right.empty:
        row = left.iloc[0]
        return float(row["lat"]), float(row["lon"])
    if left.empty:
        row = right.iloc[0]
        return float(row["lat"]), float(row["lon"])
    l = left.iloc[0]
    r = right.iloc[0]
    if float(r["timestamp"]) == float(l["timestamp"]):
        return float(l["lat"]), float(l["lon"])
    t = (timestamp_s - float(l["timestamp"])) / (float(r["timestamp"]) - float(l["timestamp"]))
    lat = float(l["lat"] + t * (r["lat"] - l["lat"]))
    lon = float(l["lon"] + t * (r["lon"] - l["lon"]))
    return lat, lon

