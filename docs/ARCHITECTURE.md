# Architecture

## Core objects

- **Lightpath**: Carries signal power, noise, latency, and bitrate along a path.
- **Line**: Models fiber spans, amplifiers, ASE/NLI noise, and channel occupancy.
- **Node**: Routes lightpaths across connected lines and optimizes launch power.
- **Network**: Builds the graph from topology, computes weighted paths, and streams traffic.
- **Connection**: Represents a traffic demand between two nodes.

## Flow

1. Load topology and instantiate nodes/lines.
2. Connect nodes/lines into a graph.
3. Build weighted paths (latency, noise, SNR).
4. Generate connections from a traffic matrix.
5. Stream connections through the network, updating route space.
6. Aggregate metrics and render plots.

## Key files

- `main.py`: All core classes and the CLI entrypoint.
- `nodes_9.json`: Sample topology.
- `docs/USAGE.md`: CLI and topology guidance.
