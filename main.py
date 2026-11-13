#!/usr/bin/env python3
"""Five-quantile, four-hour-ahead forecasts for Polymarket's "Next CEO of NVIDIA?" Yes prices.

The program is handed one case: every market of the event, its Yes price on a
one-minute grid for the 72 hours up to a checkpoint (the last price is the
price at the checkpoint).  For each market it writes, for each of the next 240
minutes, the 5%, 25%, 50%, 75% and 95% quantiles of that minute's price.

The model is an empirical h-step bootstrap, not a normal random walk:

  * each market gets its own 1-minute step scale (RMS of its minute-to-minute
    changes), because the event mixes markets that never move with markets that
    jump twenty points in a minute;
  * the growth of the h-step spread is measured from the case's own history
    (RMS of h-minute changes over the RMS of 1-minute changes) and shrunk
    half-way towards the cross-market median, so a thin or jumpy history cannot
    dictate its own spread;
  * the answer window's shape comes from the pooled distribution of historical
    h-step changes, standardised by each market's own spread, so the mass at
    exactly zero (most minutes do not move) and the fat tails are taken from
    the data rather than from a normal law.

Everything is deterministic: no randomness, no network, standard library only.
The grid is read from $MAKEFUN_INPUT_DIR (default /input) and the forecasts are
written to $MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
from pathlib import Path

# The levels the contest scores, in the order each output row lists them.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# Horizons (in minutes after the checkpoint) at which the spread ratio and the
# standardised quantiles are measured; every other minute is interpolated
# linearly in log(h) between the two surrounding anchors.
ANCHORS = (1, 2, 4, 8, 16, 32, 64, 128, 240)

# The h-step spread ratio of one market is shrunk half-way towards the median
# ratio of every market in the case.
OWN_WEIGHT = 0.5

# A market's 1-minute step scale never falls below this fraction of the median
# step scale of the case, and never below MIN_STEP.
SCALE_FLOOR_FRACTION = 0.125
MIN_STEP = 1e-05

# The empirical spread and the pooled shape both overstate how far a typical
# market travels: the step scale is a root mean square, which a single twenty
# point jump dominates, and the pooled shape mixes markets that never move with
# the few that jump every hour.  Calibrated on the kit's five practice cases
# (the only labelled windows available), these two factors minimise the pinned
# pinball loss there: the spread is taken at three quarters of the measured
# value and the 5 % / 95 % quantiles at half of the standardised shape, while
# the 25 % / 50 % / 75 % levels keep the shape as measured.
SPREAD_SHRINK = 0.75
TAIL_SHRINK = 0.5

# Guard rails so a degenerate history cannot divide by ~0 or produce a spread
# that is not in the data at all.
RATIO_LOW = 0.3
RATIO_HIGH = 60.0

# The empirical quantiles are estimated from at most this many standardised
# changes per anchor (taken with a uniform stride, never at random), which keeps
# a full 24-market case well inside the contest's 8-second budget.
MAX_SAMPLES_PER_ANCHOR = 12000
MAX_SAMPLES_PER_MARKET = 1200

# A market needs at least this many minutes of history beyond the horizon to
# measure its own h-step spread; otherwise the case's pooled ratio is used.
MIN_HISTORY_MARGIN = 20


def median(values):
    """Median of an unsorted sequence of numbers (0.0 when it is empty)."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def quantile_of_sorted(ordered, level):
    """Empirical quantile of an already-sorted list, linearly interpolated."""
    n = len(ordered)
    if n == 1:
        return ordered[0]
    position = level * (n - 1)
    low = int(math.floor(position))
    high = low + 1 if low + 1 < n else n - 1
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def step_scale(prices):
    """Root mean square of a market's 1-minute price changes."""
    n = len(prices)
    if n < 2:
        return 0.0
    total = 0.0
    previous = prices[0]
    for value in prices[1:]:
        change = value - previous
        total += change * change
        previous = value
    return math.sqrt(total / (n - 1))


def h_step_rms(prices, horizon):
    """Root mean square of the h-minute changes over the whole history."""
    n = len(prices)
    span = n - horizon
    if span <= 0:
        return None
    total = 0.0
    for index in range(span):
        change = prices[index + horizon] - prices[index]
        total += change * change
    return math.sqrt(total / span)


def interpolate_weights(horizon):
    """(anchor index, weight in log-h space) for each minute 1..horizon."""
    weights = []
    for minute in range(1, horizon + 1):
        if minute <= ANCHORS[0]:
            weights.append((0, 0.0))
            continue
        if minute >= ANCHORS[-1]:
            weights.append((len(ANCHORS) - 2, 1.0))
            continue
        upper = 1
        while ANCHORS[upper] < minute:
            upper += 1
        lower = upper - 1
        span = math.log(ANCHORS[upper]) - math.log(ANCHORS[lower])
        fraction = (math.log(minute) - math.log(ANCHORS[lower])) / span
        weights.append((lower, fraction))
    return weights


def interp(values, weights):
    """Linear-in-log-h interpolation of a per-anchor value list."""
    lower, fraction = weights
    return values[lower] * (1.0 - fraction) + values[lower + 1] * fraction


def price6(value):
    """Clamp to [0, 1] and round to six decimals (never -0.0)."""
    if value < 0.0:
        value = 0.0
    elif value > 1.0:
        value = 1.0
    return round(value, 6) + 0.0


def build_forecasts(case):
    """Forecast every market of one case: {market id -> 240 rows of 5 numbers}."""
    horizon = int(case["horizon_minutes"])
    markets = case["markets"]
    series = [list(market["prices"]) for market in markets]
    if not series:
        return {}

    scales = [max(step_scale(prices), MIN_STEP) for prices in series]
    floor = max(MIN_STEP, SCALE_FLOOR_FRACTION * median(scales))
    scales = [max(scale, floor) for scale in scales]

    # Pass 1: each market's own h-step spread ratio, and the cross-market median.
    own_ratio = []
    for horizon_h in ANCHORS:
        column = []
        for index, prices in enumerate(series):
            if len(prices) >= horizon_h + MIN_HISTORY_MARGIN:
                rms = h_step_rms(prices, horizon_h)
                column.append(rms / scales[index] if rms else None)
            else:
                column.append(None)
        own_ratio.append(column)

    pooled_ratio = []
    for column in own_ratio:
        known = [value for value in column if value is not None and value > 0.0]
        pooled_ratio.append(median(known) if known else 1.0)

    blended = []
    for anchor_index, column in enumerate(own_ratio):
        row = []
        for value in column:
            own = pooled_ratio[anchor_index] if value is None else value
            ratio = OWN_WEIGHT * own + (1.0 - OWN_WEIGHT) * pooled_ratio[anchor_index]
            row.append(min(RATIO_HIGH, max(RATIO_LOW, ratio)))
        blended.append(row)

    # Pass 2: the pooled distribution of standardised h-step changes, per anchor.
    shape = []
    for anchor_index, horizon_h in enumerate(ANCHORS):
        samples = []
        for index, prices in enumerate(series):
            span = len(prices) - horizon_h
            if span <= 0:
                continue
            denominator = scales[index] * blended[anchor_index][index]
            stride = max(1, span // MAX_SAMPLES_PER_MARKET)
            for start in range(0, span, stride):
                samples.append((prices[start + horizon_h] - prices[start]) / denominator)
        if not samples:
            samples = [0.0]
        stride = max(1, len(samples) // MAX_SAMPLES_PER_ANCHOR)
        if stride > 1:
            samples = samples[::stride]
        samples.sort()
        shape.append([quantile_of_sorted(samples, level) for level in QUANTILES])

    weights = interpolate_weights(horizon)
    ratios = [[blended[anchor_index][index] for anchor_index in range(len(ANCHORS))]
              for index in range(len(series))]
    # The two outer levels are the measured shape shrunk towards the median.
    level_shrink = (TAIL_SHRINK, 1.0, 1.0, 1.0, TAIL_SHRINK)

    forecasts = {}
    for index, market in enumerate(markets):
        prices = series[index]
        if not prices:
            raise ValueError(f"market {market.get('id')} has no prices")
        anchor = prices[-1]
        scale = scales[index] * SPREAD_SHRINK
        rows = []
        for minute in range(1, horizon + 1):
            weight = weights[minute - 1]
            spread = scale * interp(ratios[index], weight)
            levels = [interp([shape[a][k] for a in range(len(ANCHORS))], weight) * level_shrink[k]
                      for k in range(len(QUANTILES))]
            row = [price6(anchor + spread * level) for level in levels]
            for position in range(1, len(row)):
                if row[position] < row[position - 1]:
                    row[position] = row[position - 1]
            rows.append(row)
        forecasts[market["id"]] = rows
    return forecasts


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text())
        forecasts = build_forecasts(case)
        payload = {"forecasts": [{"id": key, "q": rows} for key, rows in forecasts.items()]}
        (output_dir / (path.stem + ".out")).write_text(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
