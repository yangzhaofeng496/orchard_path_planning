# Orchard-RCRA-UA Planner

A reproducible 2-D orchard navigation benchmark for a coordinated global-local planner:

- **RCRA\***: Row-Corridor Risk-Aware A* global planning.
- **Orchard UA planner**: path-tangent heading alignment, dynamic-obstacle uncertainty inflation, safe lateral refinement, and global-path stall recovery.

> Scope: simulation benchmark only. The current results do not constitute real-orchard or ROS/Nav2 field validation.

## Why this project

Orchard navigation differs from generic indoor planning because tree rows impose corridor structure, trunks require safety margins, and workers or agricultural vehicles introduce dynamic uncertainty. A generic A* + DWA stack may produce a valid global path but still stall when the local window starts with a heading misaligned with the orchard row.

## Contributions

1. **Row-corridor risk cost**: combines movement, row/corridor preference, turning cost, and obstacle-risk cost in global search.
2. **Path-tangent heading alignment**: initializes local motion from the global-path tangent to prevent zero-progress starts.
3. **Uncertainty-aware dynamic safety band**: predicts moving obstacles and laterally refines risky waypoints in collision-free space.
4. **Global-consistent stall recovery**: reconnects the local trajectory to safe global-path nodes when progress is lost.
5. **Auditable evaluation**: includes common baselines, module ablations, fixed seeds, per-episode metrics, and an automatically generated PDF report.

## Baselines

- Greedy + APF
- Dijkstra + APF
- A* + APF / DWA / Pure Pursuit
- RRT* + DWA
- Hybrid A* + TEB-inspired local planning

Hybrid A* and TEB are compact reproducible approximations in this repository, not drop-in replacements for the full ROS/Nav2 implementations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Reproduce the benchmark

```bash
python benchmark.py --episodes 16 --out results
```

Generated files:

- `results/metrics.csv`: per-episode measurements.
- `results/summary.csv`: aggregated comparison.
- `results/ablation.csv`: module ablations.
- `results/benchmark.png`: representative trajectories.
- `results/method.md`: machine-readable summary snapshot.

Generate the PDF report on macOS:

```bash
python create_report.py
```

The report script currently uses the macOS `STHeiti Medium` font. Change the font path for Linux/Windows.

## Metrics

| Metric | Meaning | Direction |
|---|---|---|
| Success rate | Fraction of episodes reaching the goal neighborhood | Higher is better |
| Mean path length | Accumulated executed local trajectory length | Lower is better among methods with top success |
| Mean collisions | Mean number of occupied-grid entries | Lower is better |
| Minimum dynamic clearance | Mean episode-wise minimum distance to moving obstacles | Higher is better among methods with top success |

Failed methods are excluded from the path-length and clearance `Best` selection to avoid rewarding early termination.

## Current result

On 16 fixed test seeds (`1000-1015`), Ours reaches **100% success with zero collisions**. On a separate check using seeds `2000-2015`, Ours also reaches **100% success**, with mean path length `42.28` and mean dynamic clearance `18.94`.

These values should be regenerated on the target machine. Do not treat them as field-performance claims.

## Repository layout

```text
.
├── benchmark.py          # simulator, baselines, Ours, evaluation
├── create_report.py      # PDF report generator
├── requirements.txt
├── README.md
└── results/              # reproducible outputs
```

## Limitations and next steps

- 2-D occupancy grids only; no localization drift or sensor noise.
- Simplified vehicle dynamics and simplified Hybrid A*/TEB baselines.
- No slope, mud, wheel slip, canopy occlusion, or real controller latency.
- Next: official ROS/Nav2 baselines, 100+ held-out scenes, confidence intervals, Gazebo/Isaac Sim, and real-orchard trials.

## Citation

This is an experimental research prototype. A formal citation will be added after the associated manuscript is finalized.
