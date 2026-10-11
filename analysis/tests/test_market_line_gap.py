"""Tests for analysis/market_line_gap.py.

The histograms are built the way core.digest builds them (nearest step, then
``clip(xi, 0, top) // width`` for a binned stat), from raw simulated runs, so
every answer can be compared with the same question asked of the raw runs.
``_oracle_at_line`` restates core.dense_pmf + leg_sim.at_line independently.
"""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_line_gap as mlg  # noqa: E402


def _digest_hist(values, width=1, scale=1, top=None):
    """pmf of a stat as the kept game stores it."""
    xi = [int(round(v * scale)) for v in values]
    if top is None:
        top = max(xi)
    steps = [min(max(k, 0), top) // width for k in xi]
    pmf = [0.0] * (top // width + 1)
    for s in steps:
        pmf[s] += 1.0 / len(steps)
    return pmf


def _oracle_at_line(pmf, line, width=1, scale=1):
    """core.dense_pmf then leg_sim.at_line, restated: (more, less, push)."""
    dense = []
    for p in pmf:
        dense.extend([p / width] * width)
    total = sum(dense)
    dense = [p / total for p in dense]
    x = line * scale
    more = sum(p for k, p in enumerate(dense) if k > x)
    less = sum(p for k, p in enumerate(dense) if k < x)
    push = sum(p for k, p in enumerate(dense) if k == x)
    return more, less, push


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


@pytest.fixture(scope="module")
def fantasy():
    rng = random.Random(13)
    return [max(round(rng.gauss(18.7, 6.0), 1), 0.0) for _ in range(40000)]


def test_whole_number_stat_matches_raw_runs_exactly(completions):
    pmf = _digest_hist(completions)
    for line in (19.5, 21.5, 22.5, 24.5):
        row = mlg.line_gap(pmf, line)
        raw_over = sum(1 for x in completions if x > line) / len(completions)
        assert row["p_over"] == pytest.approx(raw_over, abs=1e-9)
        assert row["p_over"] + row["p_under"] == pytest.approx(1.0, abs=1e-9)
        assert row["p_push"] == 0.0


def test_whole_number_line_has_a_real_push(completions):
    pmf = _digest_hist(completions)
    row = mlg.line_gap(pmf, 22)
    n = len(completions)
    raw_over = sum(1 for x in completions if x > 22) / n
    raw_under = sum(1 for x in completions if x < 22) / n
    raw_push = sum(1 for x in completions if x == 22) / n
    assert row["p_over"] == pytest.approx(raw_over, abs=1e-9)
    assert row["p_under"] == pytest.approx(raw_under, abs=1e-9)
    assert row["p_push"] == pytest.approx(raw_push, abs=1e-9)
    assert row["p_push"] > 0
    # the lean ignores the push
    assert row["p_over_no_push"] == pytest.approx(raw_over / (raw_over + raw_under))


def test_a_binned_step_holds_its_floor_value_and_the_next_four():
    # pass_yds, width 5: step 47 is 235..239 yards, mass spread evenly
    pmf = [0.0] * 114
    pmf[47] = 1.0
    assert mlg.line_gap(pmf, 234.5, width=5)["p_over"] == pytest.approx(1.0)
    assert mlg.line_gap(pmf, 239.5, width=5)["p_over"] == pytest.approx(0.0)
    inside = mlg.line_gap(pmf, 237.5, width=5)   # 238, 239 are More; 235, 236, 237 Less
    assert inside["p_over"] == pytest.approx(0.4)
    assert inside["p_under"] == pytest.approx(0.6)
    whole = mlg.line_gap(pmf, 237, width=5)      # 237 is the push
    assert whole["p_push"] == pytest.approx(0.2)
    assert whole["p_over"] == pytest.approx(0.4)
    assert whole["p_under"] == pytest.approx(0.4)
    assert whole["mean"] == pytest.approx(237.0)  # histogram mean of 235..239


def test_binned_stat_on_a_bin_edge_is_exact_and_inside_a_bin_is_close(yards):
    pmf = _digest_hist(yards, width=5)
    n = len(yards)
    # 249.5 and 244.5 are bin edges: 245-249 | 250-254, no bin is cut
    for line in (244.5, 249.5, 264.5):
        row = mlg.line_gap(pmf, line, width=5)
        raw_over = sum(1 for x in yards if x > line) / n
        assert row["p_over"] == pytest.approx(raw_over, abs=1e-9), line
    # inside a bin the sim assumes the runs are flat across it
    for line in (237.5, 247.5, 262.5):
        row = mlg.line_gap(pmf, line, width=5)
        raw_over = sum(1 for x in yards if x > line) / n
        assert row["p_over"] == pytest.approx(raw_over, abs=0.012), line
    assert mlg.line_gap(pmf, 249.5, width=5)["bin_width"] == 5


def test_scaled_stat_is_read_in_tenths(fantasy):
    pmf = _digest_hist(fantasy, scale=10)
    n = len(fantasy)
    # x = 20.05 * 10 = 200.5: More is step 201 (20.1) and up
    row = mlg.line_gap(pmf, 20.05, scale=10)
    assert row["p_over"] == pytest.approx(sum(1 for v in fantasy if v >= 20.1) / n, abs=1e-9)
    assert row["p_push"] == 0.0
    assert row["resolution"] == pytest.approx(0.1)
    # a line on a value is a push
    on = mlg.line_gap(pmf, 20.0, scale=10)
    assert on["p_push"] == pytest.approx(sum(1 for v in fantasy if v == 20.0) / n, abs=1e-9)
    assert on["p_push"] > 0
    assert row["mean"] == pytest.approx(sum(fantasy) / n, abs=0.01)


@pytest.mark.parametrize("width,scale", [(1, 1), (5, 1), (2, 1), (1, 10)])
def test_agrees_with_the_sims_own_at_line_on_random_histograms(width, scale):
    rng = random.Random(width * 100 + scale)
    for _ in range(30):
        size = rng.randint(5, 120)
        pmf = [rng.random() ** 3 if rng.random() < 0.85 else 0.0 for _ in range(size)]
        if sum(pmf) == 0:
            pmf[0] = 1.0
        top_value = size * width / scale
        lines = [round(rng.uniform(0, top_value), 2) for _ in range(6)]
        lines += [rng.randint(0, size) / scale * width // 1, rng.randint(0, size) + 0.5]
        for line in lines:
            row = mlg.line_gap(pmf, line, width, scale)
            more, less, push = _oracle_at_line(pmf, line, width, scale)
            assert row["p_over"] == pytest.approx(more, abs=1e-9), (width, scale, line)
            assert row["p_under"] == pytest.approx(less, abs=1e-9), (width, scale, line)
            assert row["p_push"] == pytest.approx(push, abs=1e-9), (width, scale, line)


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


def test_quantiles_are_ordered_and_near_the_raw_ones(yards):
    row = mlg.line_gap(_digest_hist(yards, width=5), 249.5, width=5)
    assert row["p10"] <= row["p25"] <= row["median"] <= row["p75"] <= row["p90"]
    raw = sorted(yards)
    n = len(raw)
    assert row["median"] == pytest.approx(raw[n // 2], abs=5.0)
    assert row["p25"] == pytest.approx(raw[n // 4], abs=5.0)
    assert row["p75"] == pytest.approx(raw[3 * n // 4], abs=5.0)


def test_fair_line_splits_the_runs_both_ways(completions):
    pmf = _digest_hist(completions)
    row = mlg.line_gap(pmf, 20.5)
    fair = row["fair_line"]
    assert fair % 1 == 0.5
    at_fair = mlg.line_gap(pmf, fair)
    # no neighbouring half-point line is closer to even
    for other in (fair - 1, fair + 1):
        o = mlg.line_gap(pmf, other)
        assert abs(at_fair["p_over"] - 0.5) <= abs(o["p_over"] - 0.5) + 1e-12
    assert row["fair_vs_line"] == pytest.approx(fair - 20.5)


def test_histogram_mean_reconciles_with_the_stored_mean(yards):
    pmf = _digest_hist(yards, width=5)
    stored = sum(yards) / len(yards)
    row = mlg.line_gap(pmf, 249.5, width=5, mean=stored)
    assert row["mean_source"] == "stored"
    assert row["mean"] == pytest.approx(stored)
    # flat-within-a-bin differs from the exact mean only by the in-bin slope
    assert abs(row["mean_check"]["diff"]) < 1.0


def test_wrong_scale_is_visible_in_the_mean_check(yards):
    pmf = _digest_hist(yards, width=5)
    stored = sum(yards) / len(yards)
    row = mlg.line_gap(pmf, 249.5, width=1, mean=stored)  # bins misread as 1 yard
    assert abs(row["mean_check"]["diff"]) > 100


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
        mlg.line_gap([1.0], True)
    with pytest.raises(ValueError):
        mlg.line_gap([1.0], 1.5, width=2.5)
    with pytest.raises(ValueError):
        mlg.line_gap([1.0], 1.5, scale=0)
    # NaN and negative weights are ignored, the rest renormalised
    # (the two kept weights sit at steps 2 and 3)
    row = mlg.line_gap([float("nan"), -1.0, 0.5, 0.5], 2.5)
    assert row["p_over"] == pytest.approx(0.5)
    assert row["p_under"] == pytest.approx(0.5)


def _players(completions, yards):
    return {
        "p1": {
            "name": "Test QB", "team": "HOU",
            "pmf": {"pass_cmp": _digest_hist(completions),
                    "pass_yds": _digest_hist(yards, width=5, top=568)},
            "mean": {"pass_cmp": sum(completions) / len(completions)},
            "bin": {"pass_yds": [5, 568]},
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
    assert rows[1]["bin_width"] == 5                   # width came from bin[stat][0]
    assert rows[0]["mean_source"] == "stored"
    assert rows[1]["mean_source"] == "histogram"       # no stored mean was given for it


def test_stat_line_lists_every_stat(completions, yards):
    line = mlg.stat_line(_players(completions, yards)["p1"])
    assert set(line) == {"pass_cmp", "pass_yds"}
    assert line["pass_yds"]["bin_width"] == 5
    assert line["pass_cmp"]["p25"] <= line["pass_cmp"]["p50"] <= line["pass_cmp"]["p75"]


def test_describe_states_counts_and_has_no_threshold_labels(completions):
    row = mlg.line_gap(_digest_hist(completions), 22)
    text = mlg.describe(row)
    assert "vs line 22" in text and "over" in text and "exactly on the line" in text
    for word in ("hot", "cold", "slump", "streaky", "lock"):
        assert word not in text.lower()
    assert mlg.describe({"status": "stat_not_stored"}).startswith("no simulated answer")
