"""Regression tests for the P0 fixes (fix-spec items 1-3).

Run: python -m pytest tests -q
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from pench.modules.alerting_v2 import (AlertConfigV2, DetectionPoint,  # noqa: E402
                                       evaluate_all)
from pench.modules.occupancy_safe import (overlap_report_safe,  # noqa: E402
                                          safe_home_range)
from pench.pipeline import (append_history_csv, load_alert_state,  # noqa: E402
                            load_history_csv, save_alert_state,
                            summarise_range, window_counts)

NOW = datetime(2026, 8, 15, 12, 0, 0)


def pts(n, lat=21.65, lon=79.35, tiger="T1", spread=0.01, day0=1):
    return [DetectionPoint(f"S{i % 4:02d}",
                           (NOW - timedelta(days=day0 + i)).isoformat(),
                           lat + spread * (i % 5), lon + spread * (i % 3),
                           tiger, 0.9)
            for i in range(n)]


# --------------------------------------------------------------- item 1: wiring
class TestHistoryRoundTrip:
    def test_latlon_survives_write_then_read(self, tmp_path):
        csv_path = tmp_path / "detection_history.csv"
        original = pts(6)
        append_history_csv(csv_path, original)
        loaded = load_history_csv(csv_path)
        assert len(loaded) == len(original)
        for a, b in zip(original, loaded):
            # lat stays lat and lon stays lon: no UTM round-trip
            assert b.lat == pytest.approx(a.lat, abs=1e-5)
            assert b.lon == pytest.approx(a.lon, abs=1e-5)
            assert b.tiger_id == a.tiger_id
            assert b.station_id == a.station_id

    def test_header_written_once_on_append(self, tmp_path):
        csv_path = tmp_path / "h.csv"
        append_history_csv(csv_path, pts(2))
        append_history_csv(csv_path, pts(2, day0=40))
        assert csv_path.read_text().count("tiger_id,timestamp") == 1
        assert len(load_history_csv(csv_path)) == 4

    def test_malformed_rows_are_skipped_not_fatal(self, tmp_path):
        csv_path = tmp_path / "h.csv"
        csv_path.write_text(
            "tiger_id,timestamp,station_id,lat,lon,confidence\n"
            "85,2026-08-01T10:00:00,S01,21.6,79.3,0.9\n"
            "86,2026-08-02T10:00:00,S01,not-a-number,79.3,0.9\n"
            "87,2026-08-03T10:00:00,S01,21.7,79.4,\n")
        loaded = load_history_csv(csv_path)
        assert [p.tiger_id for p in loaded] == ["85", "87"]
        assert loaded[1].confidence == 0.9      # blank confidence defaulted

    def test_missing_file_is_empty_not_error(self, tmp_path):
        assert load_history_csv(tmp_path / "nope.csv") == []


class TestAlertStatePersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "alert_state.json"
        save_alert_state(path, {"85:core_shift": 3}, NOW)
        assert load_alert_state(path) == {"85:core_shift": 3}
        assert json.loads(path.read_text())["updated"] == NOW.isoformat()

    def test_missing_and_corrupt_state_degrade_to_empty(self, tmp_path):
        assert load_alert_state(tmp_path / "none.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert load_alert_state(bad) == {}

    def test_debounce_escalates_on_second_run(self, tmp_path):
        """The whole point of persisting state: run 2 escalates, run 1 does not."""
        path = tmp_path / "alert_state.json"
        cfg = AlertConfigV2()
        base = pts(10, lat=21.65, lon=79.35, day0=45)
        cur = pts(6, lat=21.65, lon=79.50, day0=1)      # centroid shifted east
        detections = {"T1": base + cur}

        def one_run():
            alerts, keys = evaluate_all(detections, ["T1"], [],
                                        window_counts(base + cur, NOW,
                                                      cfg.current_window_days),
                                        NOW, cfg, load_alert_state(path))
            save_alert_state(path, keys, NOW)
            return {a.alert_type: a.severity for a in alerts}

        first = one_run()
        assert "core_shift" in first
        assert first["core_shift"] in ("warning", "critical")
        second = one_run()
        assert load_alert_state(path)["T1:core_shift"] == 2
        if first["core_shift"] != "critical":
            assert second["core_shift"] == "escalated"


class TestWindowCounts:
    def test_buckets_match_current_window_length(self):
        cfg = AlertConfigV2()
        # one detection every 2 days for 90 days
        points = [DetectionPoint("S01", (NOW - timedelta(days=d)).isoformat(),
                                 21.65, 79.35, "T1", 0.9)
                  for d in range(0, 90, 2)]
        counts = window_counts(points, NOW, cfg.current_window_days)
        assert len(counts["T1"]) >= 5, "activity_anomaly needs >=5 windows"
        # each historical bucket spans current_window_days, so counts are
        # comparable in magnitude to the current window's count
        current = sum(1 for p in points
                      if (NOW - datetime.fromisoformat(p.timestamp)).days
                      < cfg.current_window_days)
        assert all(abs(c - current) <= 3 for c in counts["T1"]), counts["T1"]

    def test_future_timestamps_ignored(self):
        points = [DetectionPoint("S01", (NOW + timedelta(days=5)).isoformat(),
                                 21.65, 79.35, "T1", 0.9)]
        assert window_counts(points, NOW, 30) == {}


# ------------------------------------------------- item 1: hardened home range
class TestSafeHomeRange:
    def test_centroid_is_not_lat_lon_swapped(self):
        r = safe_home_range(pts(12))["home_range"]
        assert r["quality"] in ("good", "degraded")
        assert 21.0 <= r["centroid_lat"] <= 22.5, r["centroid_lat"]
        assert 78.5 <= r["centroid_lon"] <= 80.5, r["centroid_lon"]

    def test_insufficient_data_has_no_areas(self):
        r = safe_home_range(pts(2))["home_range"]
        assert r["quality"] == "insufficient_data"
        assert "mcp_area_km2" not in r
        assert summarise_range(r)["usable"] is False

    def test_identical_points_are_degenerate_not_a_crash(self):
        same = [DetectionPoint("S01", (NOW - timedelta(days=i)).isoformat(),
                               21.65, 79.35, "T1", 0.9) for i in range(8)]
        r = safe_home_range(same)["home_range"]
        assert r["quality"] == "degenerate_geometry"
        assert summarise_range(r)["usable"] is False

    def test_out_of_reserve_coordinates_rejected(self):
        bad = [DetectionPoint("S01", (NOW - timedelta(days=i)).isoformat(),
                              0.0, 0.0, "T1", 0.9) for i in range(8)]
        r = safe_home_range(bad)["home_range"]
        assert r["n_rejected_coordinates"] == 8
        assert r["quality"] == "insufficient_data"

    def test_report_is_json_serialisable(self):
        """kde95 can come back NaN; NaN is not valid JSON for the dashboard."""
        r = safe_home_range(pts(12))["home_range"]
        assert "NaN" not in json.dumps(summarise_range(r))

    def test_mcp_polygon_is_the_full_hull_not_the_core(self):
        r = safe_home_range(pts(14))["home_range"]
        assert len(r["mcp_polygon"]) >= 4
        from pench.modules.occupancy import lonlat_to_utm
        from shapely.geometry import Polygon
        area = Polygon([lonlat_to_utm(lon, lat)
                        for lon, lat in r["mcp_polygon"]]).area / 1e6
        assert area == pytest.approx(r["mcp_area_km2"], rel=0.02)


class TestOverlapReport:
    def test_unusable_ranges_are_skipped(self):
        ranges = {"A": safe_home_range(pts(12))["home_range"],
                  "B": safe_home_range(pts(2))["home_range"]}
        assert overlap_report_safe(ranges) == {}

    def test_overlapping_tigers_reported(self):
        a = safe_home_range(pts(12, lat=21.65, lon=79.35, spread=0.02))["home_range"]
        b = safe_home_range(pts(12, lat=21.66, lon=79.36, spread=0.02))["home_range"]
        rep = overlap_report_safe({"A": a, "B": b})
        assert "A x B" in rep
        assert rep["A x B"]["overlap_km2"] > 0
        assert 0 < rep["A x B"]["pct_of_first"] <= 100


# ---------------------------------------------------- items 2 and 3: thresholds
class TestModelThresholds:
    def test_blank_filter_band_comes_from_checkpoint(self):
        torch = pytest.importorskip("torch")
        from pench.model_serving import BlankFilter, MODELS
        ckpt_path = MODELS / "blank_filter_v2.pth"
        if not ckpt_path.exists():
            pytest.skip("blank filter checkpoint not present")
        bf = BlankFilter(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert bf.band_source == "checkpoint"
        assert (bf.lo, bf.hi) == (ckpt["threshold_lo"], ckpt["threshold_hi"])
        assert bf.lo < bf.hi

    def test_blank_filter_prefers_newest_checkpoint(self):
        pytest.importorskip("torch")
        from pench.model_serving import BlankFilter, MODELS
        if not (MODELS / "blank_filter_v2.pth").exists():
            pytest.skip("blank filter checkpoint not present")
        expected = ("blank_filter_v3.pth"
                    if (MODELS / "blank_filter_v3.pth").exists()
                    else "blank_filter_v2.pth")
        assert BlankFilter().ckpt_path.name == expected

    def test_confirm_dist_matches_open_set_calibration(self):
        pytest.importorskip("torch")
        pytest.importorskip("faiss")
        from pench.model_serving import TigerReID, MODELS
        assert TigerReID.CONFIRM_DIST == pytest.approx(0.316, abs=1e-3)
        assert TigerReID.CONFIRM_DIST < TigerReID.ENROLL_DIST
        bench = MODELS / "reid_benchmark.json"
        if bench.exists():
            data = json.loads(bench.read_text())
            best = (data.get("task3", {})
                        .get("calibrated_distance_threshold", {}).get("best_t"))
            if best is not None:
                assert TigerReID.CONFIRM_DIST == pytest.approx(best, abs=5e-3)

    def test_alert_config_tracks_reid_threshold(self):
        pytest.importorskip("torch")
        pytest.importorskip("faiss")
        from pench.model_serving import TigerReID
        assert (AlertConfigV2().new_identity_min_distance
                == pytest.approx(TigerReID.CONFIRM_DIST))
