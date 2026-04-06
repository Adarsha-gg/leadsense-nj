from __future__ import annotations

import pandas as pd

from leadsense_nj.satellite import (
    aggregate_sentinel_tile_features,
    build_bbox_from_point,
    fetch_sentinel_features_for_block_groups,
    validate_sentinel_feature_frame,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def post(self, url, json=None, timeout=None):  # noqa: ANN001
        _ = (url, timeout)
        bbox = json.get("bbox", [])
        lon = (bbox[0] + bbox[2]) / 2.0
        lat = (bbox[1] + bbox[3]) / 2.0
        feature = {
            "id": f"S2_FAKE_{lat:.3f}_{lon:.3f}",
            "properties": {
                "datetime": "2024-06-15T12:00:00Z",
                "eo:cloud_cover": 12.0,
                "s2:vegetation_percentage": 36.0,
                "s2:water_percentage": 5.0,
                "s2:nodata_pixel_percentage": 1.5,
                "view:sun_elevation": 57.0,
            },
        }
        return _FakeResponse({"type": "FeatureCollection", "features": [feature]})


def test_build_bbox_from_point_shape() -> None:
    bbox = build_bbox_from_point(40.1, -74.5, half_size_deg=0.02)
    assert len(bbox) == 4
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


def test_aggregate_sentinel_tile_features_empty_fallback() -> None:
    agg = aggregate_sentinel_tile_features([])
    assert agg["s2_item_count"] == 0.0
    assert agg["s2_cloud_cover_mean"] >= 0.0


def test_fetch_sentinel_features_for_block_groups_with_fake_session() -> None:
    df = pd.DataFrame({"geoid": ["340010001001", "340030010002"], "lat": [40.1, 40.2], "lon": [-74.5, -74.6]})
    out = fetch_sentinel_features_for_block_groups(
        df,
        start_date="2024-05-01",
        end_date="2024-08-01",
        items_per_block=1,
        session=_FakeSession(),
        allow_fallback=False,
    )
    validate_sentinel_feature_frame(out)
    assert len(out) == 2
    assert set(out["s2_source"]) == {"live"}
