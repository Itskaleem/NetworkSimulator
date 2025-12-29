# NetworkSimulator

A Python network infrastructure simulator focused on optical lightpath routing, latency/SNR optimization, and Monte Carlo traffic studies.

## Features

- Optical line propagation with ASE and NLI noise modeling.
- Latency and SNR-weighted routing for traffic demands.
- Monte Carlo simulation loop with congestion analysis.
- Configurable traffic matrix generation and transceiver modes.
- Sample 9-node topology included (`nodes_9.json`).

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `matplotlib`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run the default simulation:

```bash
python main.py
```

Customize topology and simulation parameters:

```bash
python main.py \
  --topology nodes_9.json \
  --channels 10 \
  --iterations 5 \
  --upgrade-line DB \
  --best snr \
  --transceiver shannon
```

## Output

The script prints summary statistics and displays plots for:

- Line congestion
- SNR distributions

See [`docs/USAGE.md`](docs/USAGE.md) for deeper usage and topology guidance.

## Project layout

- `main.py`: core simulation logic and CLI.
- `nodes_9.json`: sample topology.
- `docs/`: extended documentation.
