"""Tests for analysis/market_line_gap.py.

The histograms here are built from raw simulated runs, so every answer can be
compared with the same question asked of the raw runs.
"""
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_line_gap as mlg  # noqa: E402


def _hist(values, step=1.0):
    """The digest's histogram: step i holds the value i * step (nearest step)."""
    idx = [max(int(round(v / step)), 0) for v in values]
    pmf = [0.0] * (max(idx) + 1)
    for i in idx:
        pmf[i] += 1.0 / len(idx)
    return pmf


def _binomial(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


@pytest.fixture(scope="module")
def completions():
    rng = random.Random(7)
    return [_binomial(rng, 32, 0.68) for _ in range(40000)]


@pytest.fixture(scope="module")
def yards():
    rng = random.Random(11)
    return [max(int(round(rng.gauss(240, 60))), 0) for _ in range(40000)]


def test_whole_number_stat_matches_raw_runs_exactly(completions):
    pmf = _hist(completions)
    for line in (19.5, 21.5, 22.5, 24.5):
        row = mlg.line_gap(pmf, line)
        raw_over = sum(1 for x in completions if x > line) / len(completions)
        assert row["p_over"] == pytest.approx(raw_over, abs=1e-9)
        assert row["p_over"] + row["p_under"] == pytest.approx(1.0, abs=1e-9)
        assert row["p_push"] == 0.0
        assert row["push_resolved"] is True


def test_whole_number_line_has_a_real_push(completions):
    pmf = _hist(completions)
    row = mlg.line_gap(pmf, 22)
    raw = {
        "over": sum(1 for x in completions if x > 22) / len(completions),
        "under": sum(1 for x in completions if x < 22) / len(completions),
        "push": sum(1 for x in completions if x == 22) / len(completions),
    }
    assert row["p_over"] == pytest.approx(raw["over"], abs=1e-9)
    assert row["p_under"] == pytest.approx(raw["under"], abs=1e-9)
    assert row["p_push"] == pytest.approx(raw["push"], abs=1e-9)
    assert row["p_push"] > 0
    # the lean ignores the push
    assert row["p_over_no_push"] == pytest.approx(raw["over"] / (raw["over"] + raw["under"]))


def test_binned_stat_close_to_raw_runs_at_a_line_inside_a_bin(yards):
    pmf = _hist(yards, step=5.0)
    for line in (199.5, 237.5, 249.5, 262.5, 299.5):
        row = mlg.line_gap(pmf, line, width=5)
        raw_over = sum(1 for x in yards if x > line) / len(yards)
        # splitting a 5-yard bin in proportion is exact only if the bin is flat
        assert row["p_over"] == pytest.approx(raw_over, abs=0.012), line
        assert row["resolution"] == 5.0
        assert row["push_resolved"] is False
        assert row["p_push"] == 0.0


def test_scaled_stat_is_read_in_tenths():
    # fantasy points are stored in tenths: width 1, scale 10
    values = [round(x * 0.1, 1) for x in range(100, 301)]  # 10.0 .. 30.0
    pmf = _hist(values, step=0.1)
    row = mlg.line_gap(pmf, 20.05, width=1, scale=10)
    assert row["resolution"] == pytest.approx(0.1)
    assert row["p_over"] == pytest.approx(0.5, abs=0.01)
    assert row["mean"] == pytest.approx(20.0, abs=0.01)


def test_gap_and_averages():
    # half the runs at 0, half at 10; line 4.5
    pmf = [0.5] + [0.0] * 9 + [0.5]
    row = mlg.line_gap(pmf, 4.5)
    assert row["mean"] == pytest.approx(5.0)
    assert row["gap"] == pytest.approx(0.5)
    assert row["gap_pct"] == pytest.approx(0.5 / 4.5)
    assert row["p_over"] == pytest.approx(0.5)
    assert row["avg_over_by"] == pytest.approx(5.5)   # 10 - 4.5
    assert row["avg_under_by"] == pytest.approx(4.5)  # 4.5 - 0
    assert row["lean"] is None  # exactly even: no side is named


def test_mean_and_chance_can_disagree_and_both_are_reported():
    # 90% of runs at 0, 10% at 100: the average is above 5.5, most runs are below
    pmf = [0.9] + [0.0] * 99 + [0.1]
    row = mlg.line_gap(pmf, 5.5)
    assert row["mean"] == pytest.approx(10.0)
    assert row["gap"] > 0
    assert row["lean"] == "UNDER"
    assert row["mean_side"] == "OVER"
    assert row["mean_agrees_with_lean"] is False
    assert row["median"] == 0.0 and row["median_gap"] < 0


def test_quantiles_are_ordered(yards):
    row = mlg.line_gap(_hist(yards, step=5.0), 249.5, width=5)
    assert row["p10"] <= row["p25"] <= row["median"] <= row["p75"] <= row["p90"]
    raw = sorted(yards)
    assert row["median"] == pytest.approx(raw[len(raw) // 2], abs=3.0)


def test_fair_line_splits_the_runs_both_ways(completions):
    pmf = _hist(completions)
    row = mlg.line_gap(pmf, 20.5)
    fair = row["fair_line"]
    assert fair % 1 == 0.5
    at_fair = mlg.line_gap(pmf, fair)
    assert abs(at_fair["p_over"] - 0.5) < 0.1
    # no neighbouring half-point line is closer to even
    for other in (fair - 1, fair + 1):
        o = mlg.line_gap(pmf, other)
        assert abs(at_fair["p_over"] - 0.5) <= abs(o["p_over"] - 0.5) + 1e-12
    assert row["fair_vs_line"] == pytest.approx(fair - 20.5)


def test_histogram_mean_reconciles_with_the_stored_mean(yards):
    pmf = _hist(yards, step=5.0)
    stored = sum(yards) / len(yards)
    row = mlg.line_gap(pmf, 249.5, width=5, mean=stored)
    assert row["mean_source"] == "stored"
    assert row["mean"] == pytest.approx(stored)
    # a wrong reading of the steps would show up here
    assert abs(row["mean_check"]["diff"]) < 1.0


def test_wrong_scale_is_visible_in_the_mean_check(yards):
    pmf = _hist(yards, step=5.0)
    stored = sum(yards) / len(yards)
    row = mlg.line_gap(pmf, 249.5, width=1, mean=stored)  # steps misread as 1 yard
    assert abs(row["mean_check"]["diff"]) > 100


def test_floor_anchor_shifts_a_binned_stat_by_half_a_bin(yards):
    pmf = _hist(yards, step=5.0)
    centre = mlg.line_gap(pmf, 249.5, width=5, anchor="centre")
    floor = mlg.line_gap(pmf, 249.5, width=5, anchor="floor")
    assert floor["p_over"] > centre["p_over"]


def test_bad_inputs():
    with pytest.raises(ValueError):
        mlg.line_gap([], 1.5)
    with pytest.raises(ValueError):
        mlg.line_gap([0.0, 0.0], 1.5)
    with pytest.raises(ValueError):
        mlg.line_gap([1.0], float("nan"))
    with pytest.raises(ValueError):
        mlg.line_gap([1.0], None)
    with pytest.raises(ValueError):
        mlg.line_gap([1.0], 1.5, anchor="middle")
    # NaN and negative weights are ignored, the rest renormalised
    # (the two kept weights sit at steps 2 and 3)
    row = mlg.line_gap([float("nan"), -1.0, 0.5, 0.5], 2.5)
    assert row["p_over"] == pytest.approx(0.5)
    assert row["p_under"] == pytest.approx(0.5)


def _players(completions, yards):
    return {
        "p1": {
            "name": "Test QB", "team": "HOU",
            "pmf": {"pass_cmp": _hist(completions), "pass_yds": _hist(yards, 5.0)},
            "mean": {"pass_cmp": sum(completions) / len(completions)},
            "bin": {"pass_yds": [5, 255]},
        },
    }


def test_gap_rows_keeps_every_line_and_says_why_one_has_no_answer(completions, yards):
    players = _players(completions, yards)
    lines = [
        {"pkey": "p1", "stat": "pass_cmp", "line": 20.5, "book": "prizepicks"},
        {"pkey": "p1", "stat": "pass_yds", "line": 249.5, "book": "underdog"},
        {"pkey": "p1", "stat": "rush_yds", "line": 18.5, "book": "betr"},
        {"pkey": "zz", "stat": "pass_cmp", "line": 20.5, "book": "betr"},
        {"pkey": "p1", "stat": "pass_cmp", "line": float("nan"), "book": "x"},
    ]
    rows = mlg.gap_rows(players, lines)
    assert len(rows) == len(lines)
    assert [r["status"] for r in rows[:4]] == ["ok", "ok", "stat_not_stored", "no_simulation"]
    assert rows[4]["status"].startswith("bad_input")
    assert rows[0]["book"] == "prizepicks"            # extras pass through
    assert rows[1]["resolution"] == 5.0                # width came from bin[stat]
    assert rows[0]["mean_source"] == "stored"


def test_stat_line_lists_every_stat(completions, yards):
    line = mlg.stat_line(_players(completions, yards)["p1"])
    assert set(line) == {"pass_cmp", "pass_yds"}
    assert line["pass_yds"]["resolution"] == 5.0
    assert line["pass_cmp"]["p25"] <= line["pass_cmp"]["p50"] <= line["pass_cmp"]["p75"]


def test_describe_states_counts_and_has_no_threshold_labels(completions):
    row = mlg.line_gap(_hist(completions), 22)
    text = mlg.describe(row)
    assert "vs line 22" in text and "over" in text and "exactly on the line" in text
    for word in ("hot", "cold", "slump", "streaky", "lock"):
        assert word not in text.lower()
    assert mlg.describe({"status": "stat_not_stored"}).startswith("no simulated answer")
