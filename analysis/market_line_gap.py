"""Where the simulation's projection sits against a posted market line.

Reads the per-player distributions the kept games already store
(``digest["players"][pkey]`` in ``cache/game_sim/<game>.json``)::

    {"name": ..., "team": ...,
     "pmf":   {stat: [P(step 0), P(step 1), ...]},   # 40,000 runs as a histogram
     "mean":  {stat: exact mean over the runs},
     "bin":   {stat: [width, top]},                  # optional: width defaults to 1
     "scale": {stat: 10}}                            # optional: defaults to 1

How a step is read (this is ``core.digest`` / ``core.dense_pmf`` in the sim):

* ordinary stat (width 1, scale 1): step ``i`` is the value ``i``.
* binned stat (width ``w``): step ``i`` holds the whole-number values
  ``i*w .. i*w + w - 1``; its mass is spread evenly over those values.
  (Passing yards, width 5: step 47 is 235 to 239 yards.)
* scaled stat (``scale`` 10, fantasy points): step ``i`` is the value
  ``i / scale``, to the nearest tenth.

So every stat is a set of exact values on a lattice, and a line is compared
with them the way ``leg_sim.at_line`` does: More is strictly above the line,
Less is strictly below, and a line sitting exactly on a value is a push.
Nothing is fitted or corrected here.

``line_gap`` turns one stat's histogram and one posted line into the figures a
card shows; ``stat_line`` summarises every stat of one player; ``gap_rows``
runs a list of posted lines over a game's players and keeps a row for every
line it was given (a line it cannot answer says why).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_EPS = 1e-9


class Dist:
    """One stat's histogram as exact values on a lattice."""

    def __init__(self, pmf: Sequence[float], width: float = 1, scale: float = 1) -> None:
        weights = [float(p) if (p is not None and p == p and p > 0) else 0.0 for p in pmf]
        total = sum(weights)
        if not total > 0:
            raise ValueError("empty distribution")
        if int(width) != width or width < 1:
            raise ValueError("bin width must be a whole number of at least 1")
        if not scale > 0:
            raise ValueError("scale must be positive")
        self.width = int(width)
        self.scale = float(scale)
        # lattice index k (value = k / scale) and its probability, in order
        self.ks: List[int] = []
        self.ps: List[float] = []
        for i, w in enumerate(weights):
            if w <= 0:
                continue
            p = w / total
            if self.width == 1:
                self.ks.append(i)
                self.ps.append(p)
            else:
                for j in range(self.width):
                    self.ks.append(i * self.width + j)
                    self.ps.append(p / self.width)
        self.mean = sum(p * k for k, p in zip(self.ks, self.ps)) / self.scale

    @property
    def resolution(self) -> float:
        """Spacing between the values the stat can take."""
        return 1.0 / self.scale

    # -- reading the distribution -------------------------------------------------

    def quantile(self, q: float) -> float:
        """The first value with share ``q`` of the runs at or below it."""
        cum = 0.0
        for k, p in zip(self.ks, self.ps):
            cum += p
            if cum >= q - 1e-12:
                return k / self.scale
        return self.ks[-1] / self.scale

    def sides(self, line: float) -> Tuple[float, float, float, Optional[float], Optional[float]]:
        """Return ``(over, under, push, mean_when_over, mean_when_under)`` at ``line``."""
        x = line * self.scale
        over = under = push = 0.0
        over_sum = under_sum = 0.0
        for k, p in zip(self.ks, self.ps):
            if k > x + _EPS:
                over += p
                over_sum += p * k
            elif k < x - _EPS:
                under += p
                under_sum += p * k
            else:
                push += p
        return (
            over, under, push,
            over_sum / over / self.scale if over > 0 else None,
            under_sum / under / self.scale if under > 0 else None,
        )

    def fair_line(self) -> float:
        """The line at which More and Less are equally likely.

        A whole-number stat gets the nearest half-point line (the posted kind);
        a scaled stat (tenths) gets its median.
        """
        median = self.quantile(0.5)
        if self.scale != 1.0:
            return median
        centre = round(median)
        best: Optional[Tuple[float, float, float]] = None
        for line in (centre - 1.5, centre - 0.5, centre + 0.5, centre + 1.5):
            over, under, _push, _a, _b = self.sides(line)
            key = (abs(over - under), abs(line - self.mean), line)
            if best is None or key < best:
                best = key
        assert best is not None
        return best[2]


def line_gap(pmf: Sequence[float], line: float, width: float = 1, scale: float = 1,
             mean: Optional[float] = None) -> Dict[str, object]:
    """How the simulated stat sits against one posted line.

    ``mean`` is the exact stored mean when there is one; the histogram's own
    mean is kept alongside it so a card (or a test) can see they agree.
    """
    if (line is None or isinstance(line, bool) or not isinstance(line, (int, float))
            or not math.isfinite(line)):
        raise ValueError("line must be a finite number")
    dist = Dist(pmf, width, scale)
    over, under, push, over_mean, under_mean = dist.sides(line)
    used = float(mean) if mean is not None else dist.mean
    gap = used - line
    median = dist.quantile(0.5)
    decided = over + under
    p_over_np = over / decided if decided > 0 else None
    lean = None
    if p_over_np is not None and p_over_np != 0.5:
        lean = "OVER" if p_over_np > 0.5 else "UNDER"
    mean_side = "OVER" if gap > 0 else "UNDER" if gap < 0 else None
    fair = dist.fair_line()
    return {
        "line": float(line),
        "mean": used,
        "mean_source": "stored" if mean is not None else "histogram",
        "gap": gap,
        "gap_pct": gap / line if line else None,
        "median": median,
        "median_gap": median - line,
        "p10": dist.quantile(0.10),
        "p25": dist.quantile(0.25),
        "p75": dist.quantile(0.75),
        "p90": dist.quantile(0.90),
        "p_over": over,
        "p_under": under,
        "p_push": push,
        "p_over_no_push": p_over_np,
        "lean": lean,
        "mean_side": mean_side,
        "mean_agrees_with_lean": (mean_side == lean) if (lean and mean_side) else None,
        "avg_over_by": (over_mean - line) if over_mean is not None else None,
        "avg_under_by": (line - under_mean) if under_mean is not None else None,
        "fair_line": fair,
        "fair_vs_line": fair - line,
        "resolution": dist.resolution,
        "bin_width": dist.width,
        "mean_check": {
            "histogram": dist.mean,
            "stored": float(mean) if mean is not None else None,
            "diff": (dist.mean - float(mean)) if mean is not None else None,
        },
    }


def _stat_args(player: dict, stat: str) -> Tuple[float, float]:
    width = 1
    pair = (player.get("bin") or {}).get(stat)
    if pair:
        width = pair[0]  # bin[stat] = [width, top]; top is the highest value kept
    scale = (player.get("scale") or {}).get(stat, 1)
    return width, scale


def stat_line(player: dict) -> Dict[str, Dict[str, object]]:
    """Every stored stat of one player: mean, middle range and resolution."""
    out: Dict[str, Dict[str, object]] = {}
    for stat, pmf in (player.get("pmf") or {}).items():
        try:
            width, scale = _stat_args(player, stat)
            dist = Dist(pmf, width, scale)
        except ValueError:
            continue
        stored = (player.get("mean") or {}).get(stat)
        out[stat] = {
            "mean": float(stored) if stored is not None else dist.mean,
            "p10": dist.quantile(0.10),
            "p25": dist.quantile(0.25),
            "p50": dist.quantile(0.50),
            "p75": dist.quantile(0.75),
            "p90": dist.quantile(0.90),
            "resolution": dist.resolution,
            "bin_width": dist.width,
            "mean_check_diff": (dist.mean - float(stored)) if stored is not None else None,
        }
    return out


def gap_rows(players: Dict[str, dict], lines: Iterable[dict]) -> List[Dict[str, object]]:
    """One output row per posted line, in the order given, none dropped.

    Each ``lines`` entry carries ``pkey`` (the digest key), ``stat`` (the
    digest's stat key) and ``line``; anything else on it is passed through.
    ``status`` is ``ok`` or says why the line has no simulated answer.
    """
    rows: List[Dict[str, object]] = []
    for entry in lines:
        row: Dict[str, object] = dict(entry)
        player = players.get(entry.get("pkey"))
        stat = entry.get("stat")
        if player is None:
            row["status"] = "no_simulation"
        elif stat not in (player.get("pmf") or {}):
            row["status"] = "stat_not_stored"
        else:
            try:
                width, scale = _stat_args(player, stat)
                row.update(line_gap(player["pmf"][stat], entry.get("line"), width, scale,
                                    (player.get("mean") or {}).get(stat)))
                row["status"] = "ok"
            except ValueError as exc:
                row["status"] = "bad_input: %s" % exc
        rows.append(row)
    return rows


def describe(row: Dict[str, object], unit: str = "") -> str:
    """One plain sentence of counts and figures; no labels chosen by a threshold."""
    status = row.get("status", "ok")  # a bare line_gap() result has no status: it answered
    if status != "ok" or "gap" not in row:
        return "no simulated answer (%s)" % status
    gap = float(row["gap"])
    word = "over" if gap > 0 else "under"
    amount = "%.1f%s %s the line" % (abs(gap), unit, word) if gap else "level with the line"
    text = "projects %.1f%s vs line %g: %s; %.1f%% of runs over, %.1f%% under" % (
        float(row["mean"]), unit, float(row["line"]), amount,
        100 * float(row["p_over"]), 100 * float(row["p_under"]))
    if float(row["p_push"]) > 0:
        text += ", %.1f%% exactly on the line" % (100 * float(row["p_push"]))
    return text
