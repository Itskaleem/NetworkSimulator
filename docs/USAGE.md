# Usage

## Running simulations

The simulator uses a topology JSON file that defines node positions and connectivity. The default `nodes_9.json` file ships with a 3x3 grid of nodes.

```bash
python main.py --topology nodes_9.json
```

### Key parameters

- `--channels`: Number of WDM channels per line.
- `--iterations`: Monte Carlo iterations.
- `--upgrade-line`: Line label to model a 3 dB noise figure improvement.
- `--best`: Routing objective (`latency` or `snr`).
- `--transceiver`: `shannon`, `fixed-rate`, or `flex-rate`.
- `--traffic-rate` / `--traffic-multiplier`: Control the traffic matrix values.

### Example

```bash
python main.py \
  --topology nodes_9.json \
  --channels 8 \
  --iterations 3 \
  --upgrade-line EH \
  --best latency \
  --transceiver flex-rate
```

## Topology format

Each node entry defines:

- `position`: `[x, y]` coordinates (used for line lengths).
- `connected_nodes`: list of node labels with bidirectional links.

Example snippet:

```json
{
  "A": {
    "position": [0, 0],
    "connected_nodes": ["B", "D"]
  }
}
```

Ensure that links are bidirectional (if `A` lists `B`, then `B` should list `A`).
