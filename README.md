# Five-quantile forecasts for Polymarket's "Next CEO of NVIDIA?" markets

Entry for make.fun's `polymarket-nvidia-ceo-forecast-01`: for every market of the
event, forecast the Yes price at each of the 240 minutes after a checkpoint as
five quantiles (5 %, 25 %, 50 %, 75 %, 95 %).

`main.py` (Python 3.13, standard library only) reads every `*.in` case from
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

## Model

An empirical h-step bootstrap with a per-market scale:

1. each market's 1-minute step scale is the root mean square of its price
   changes, floored at an eighth of the case's median scale;
2. the growth of the h-minute spread is measured in the case's own history
   (RMS of h-step changes over the 1-minute scale) at nine anchors from 1 to
   240 minutes, shrunk half-way towards the cross-market median, and
   interpolated linearly in log h;
3. the answer window's shape is the pooled distribution of standardised h-step
   changes — which is where the mass at exactly zero (most minutes do not move)
   and the fat tails come from, instead of a normal law;
4. two calibration factors, measured on the kit's five practice windows,
   correct the over-dispersion of a root-mean-square scale and of a distribution
   pooled across markets that never move and markets that jump: the spread is
   taken at three quarters of the measured value and the outer (5 % / 95 %)
   levels at half of the standardised shape.

The median stays at the checkpoint price: a drift fitted to the market's own
history made every practice case worse. Everything is deterministic — no
randomness, no network, no state — and a full 24-market case runs in about a
fifth of a second.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
