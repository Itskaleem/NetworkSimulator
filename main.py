"""Network simulator for optical lightpath routing and capacity analysis."""

from __future__ import annotations

import argparse
import copy
import itertools as it
import json
from dataclasses import dataclass, field
from random import shuffle
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import Planck, c, pi
from scipy.special import erfcinv


@dataclass
class Lightpath:
    path: str
    channel: int = 0
    rs: float = 32e9
    df: float = 50e9
    transceiver: str = "shannon"
    signal_power: Optional[float] = None
    noise_power: float = 0
    snr: Optional[float] = None
    latency: float = 0
    optimized_powers: Dict[str, float] = field(default_factory=dict)
    bitrate: Optional[float] = None

    def update_snr(self, snr: float) -> None:
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)

    def add_noise(self, noise: float) -> None:
        self.noise_power += noise

    def add_latency(self, latency: float) -> None:
        self.latency += latency

    def next(self) -> None:
        self.path = self.path[1:]


class Node:
    def __init__(self, node_dict: Dict[str, object]):
        self._label = node_dict["label"]
        self._position = node_dict["position"]
        self._connected_nodes = node_dict["connected_nodes"]
        self._successive: Dict[str, Line] = {}

    @property
    def label(self) -> str:
        return self._label

    @property
    def position(self) -> Sequence[float]:
        return self._position

    @property
    def connected_nodes(self) -> Sequence[str]:
        return self._connected_nodes

    @property
    def successive(self) -> Dict[str, "Line"]:
        return self._successive

    @successive.setter
    def successive(self, successive: Dict[str, "Line"]) -> None:
        self._successive = successive

    def propagate(self, lightpath: Lightpath, occupation: bool = False) -> Lightpath:
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    def optimize(self, lightpath: Lightpath) -> Lightpath:
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            ase = line.ase_generation()
            eta = line.nli_generation(1, lightpath.rs, lightpath.df)
            p_opt = (ase / (2 * eta)) ** (1 / 3)
            lightpath.optimized_powers.update({line_label: p_opt})
            lightpath.next()
            node = line.successive[lightpath.path[0]]
            lightpath = node.optimize(lightpath)
        return lightpath


class Line:
    def __init__(self, line_dict: Dict[str, object]):
        self._label = line_dict["label"]
        self._length = line_dict["length"] * 1e3
        self._amplifiers = int(np.ceil(self._length / 80e3))
        self._span_length = self._length / self._amplifiers
        self._noise_figure = 7
        self._alpha = 4.6e-5
        self._beta = 6.27e-27
        self._gamma = 1.27e-3
        self._Nch = line_dict["Nch"]
        self._state = ["free"] * self._Nch
        self._successive: Dict[str, Node] = {}
        self._gain = self.transparency()

    @property
    def label(self) -> str:
        return self._label

    @property
    def length(self) -> float:
        return self._length

    @property
    def state(self) -> List[str]:
        return self._state

    @state.setter
    def state(self, state: Sequence[str]) -> None:
        normalized_state = [s.lower().strip() for s in state]
        if set(normalized_state).issubset({"free", "occupied"}):
            self._state = list(normalized_state)
        else:
            print(
                "ERROR: line state  not recognized."
                f"Value: {set(normalized_state) - {'free', 'occupied'}}"
            )

    @property
    def amplifiers(self) -> int:
        return self._amplifiers

    @property
    def span_length(self) -> float:
        return self._span_length

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, gain: float) -> None:
        self._gain = gain

    @property
    def Nch(self) -> int:
        return self._Nch

    @property
    def noise_figure(self) -> float:
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, noise_figure: float) -> None:
        self._noise_figure = noise_figure

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def successive(self) -> Dict[str, Node]:
        return self._successive

    @successive.setter
    def successive(self, successive: Dict[str, Node]) -> None:
        self._successive = successive

    def transparency(self) -> float:
        return 10 * np.log10(np.exp(self.alpha * self.span_length))

    def latency_generation(self) -> float:
        return self.length / (c * 2 / 3)

    def noise_generation(self, lightpath: Lightpath) -> float:
        return self.ase_generation() + self.nli_generation(
            lightpath.signal_power, lightpath.rs, lightpath.df
        )

    def ase_generation(self) -> float:
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        f = 193.4e12
        Bn = 12.5e9
        return self._amplifiers * Planck * f * Bn * noise_figure_lin * (gain_lin - 1)

    def nli_generation(self, signal_power: float, Rs: float, df: float, Bn: float = 12.5e9) -> float:
        loss = np.exp(-self.alpha * self.span_length)
        gain_lin = 10 ** (self.gain / 10)
        eta = (
            16
            / (27 * pi)
            * self.gamma**2
            / (4 * self.alpha * self.beta * Rs**3)
            * np.log(
                pi**2 * self.beta * Rs**2 * self.Nch ** (2 * Rs / df) / (2 * self.alpha)
            )
        )
        return self._amplifiers * (signal_power**3 * loss * gain_lin * eta * Bn)

    def propagate(self, lightpath: Lightpath, occupation: bool = False) -> Lightpath:
        lightpath.add_latency(self.latency_generation())
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = "occupied"
            self.state = new_state
        node = self.successive[lightpath.path[0]]
        return node.propagate(lightpath, occupation)


class Network:
    def __init__(self, json_path: str, nch: int = 10, upgrade_line: str = ""):
        node_json = json.load(open(json_path, "r"))
        self._nodes: Dict[str, Node] = {}
        self._lines: Dict[str, Line] = {}
        self._connected = False
        self._weighted_paths: Optional[pd.DataFrame] = None
        self._route_space: Optional[pd.DataFrame] = None
        self._Nch = nch
        self._upgrade_line = upgrade_line
        for node_label in node_json:
            node_dict = node_json[node_label]
            node_dict["label"] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            for connected_node_label in node_dict["connected_nodes"]:
                line_dict: Dict[str, object] = {}
                line_label = node_label + connected_node_label
                line_dict["label"] = line_label
                node_position = np.array(node_json[node_label]["position"])
                connected_node_position = np.array(node_json[connected_node_label]["position"])
                line_dict["length"] = np.sqrt(
                    np.sum((node_position - connected_node_position) ** 2)
                )
                line_dict["Nch"] = self.Nch
                line = Line(line_dict)
                self._lines[line_label] = line
        if upgrade_line:
            self.lines[self._upgrade_line].noise_figure = (
                self.lines[upgrade_line].noise_figure - 3
            )

    @property
    def Nch(self) -> int:
        return self._Nch

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    @property
    def lines(self) -> Dict[str, Line]:
        return self._lines

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def weighted_paths(self) -> Optional[pd.DataFrame]:
        return self._weighted_paths

    @property
    def route_space(self) -> Optional[pd.DataFrame]:
        return self._route_space

    def draw(self) -> None:
        for node_label, node in self.nodes.items():
            x0, y0 = node.position
            plt.plot(x0, y0, "go", markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)
            for connected_node_label in node.connected_nodes:
                connected_node = self.nodes[connected_node_label]
                x1, y1 = connected_node.position
                plt.plot([x0, x1], [y0, y1], "b")
        plt.title("Network")
        plt.show()

    def find_paths(self, label1: str, label2: str) -> List[str]:
        cross_nodes = [
            key for key in self.nodes.keys() if ((key != label1) & (key != label2))
        ]
        cross_lines = self.lines.keys()
        inner_paths = {"0": label1}
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [
                    inner_path + cross_node
                    for cross_node in cross_nodes
                    if (
                        (inner_path[-1] + cross_node in cross_lines)
                        and (cross_node not in inner_path)
                    )
                ]
        paths: List[str] = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self) -> None:
        for node_label, node in self.nodes.items():
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = self.lines[line_label]
                line.successive[connected_node] = self.nodes[connected_node]
                node.successive[line_label] = line
        self._connected = True

    def propagate(self, lightpath: Lightpath, occupation: bool = False) -> Lightpath:
        start_node = self.nodes[lightpath.path[0]]
        return start_node.propagate(lightpath, occupation)

    def optimization(self, lightpath: Lightpath) -> Lightpath:
        path = lightpath.path
        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)
        optimized_lightpath.path = path
        return optimized_lightpath

    def set_weighted_paths(self) -> None:
        if not self.connected:
            self.connect()
        node_labels = self.nodes.keys()
        pairs = [label1 + label2 for label1 in node_labels for label2 in node_labels if label1 != label2]

        paths: List[str] = []
        latencies: List[float] = []
        noises: List[float] = []
        snrs: List[float] = []

        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = "->".join(path)
                paths.append(path_string)

                lightpath = Lightpath(path)
                lightpath = self.optimization(lightpath)
                lightpath = self.propagate(lightpath, occupation=False)

                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(10 * np.log10(lightpath.signal_power / lightpath.noise_power))

        self._weighted_paths = pd.DataFrame(
            {"path": paths, "latency": latencies, "noise": noises, "snr": snrs}
        )

        route_space = pd.DataFrame({"path": paths})
        for i in range(self.Nch):
            route_space[str(i)] = ["free"] * len(paths)
        self._route_space = route_space

    def available_paths(self, input_node: str, output_node: str) -> List[str]:
        if self.weighted_paths is None:
            self.set_weighted_paths()
        assert self.weighted_paths is not None
        assert self.route_space is not None
        all_paths = [
            path
            for path in self.weighted_paths.path.values
            if ((path[0] == input_node) and (path[-1] == output_node))
        ]
        available_paths = []
        for path in all_paths:
            path_occupancy = self.route_space.loc[
                self.route_space.path == path
            ].T.values[1:]
            if "free" in path_occupancy:
                available_paths.append(path)
        return available_paths

    def find_best_snr(self, input_node: str, output_node: str) -> Optional[str]:
        available_paths = self.available_paths(input_node, output_node)
        if not available_paths:
            return None
        assert self.weighted_paths is not None
        inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
        best_snr = np.max(inout_df.snr.values)
        return inout_df.loc[inout_df.snr == best_snr].path.values[0]

    def find_best_latency(self, input_node: str, output_node: str) -> Optional[str]:
        available_paths = self.available_paths(input_node, output_node)
        if not available_paths:
            return None
        assert self.weighted_paths is not None
        inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
        best_latency = np.min(inout_df.latency.values)
        return inout_df.loc[inout_df.latency == best_latency].path.values[0]

    def stream(
        self, connections: Iterable["Connection"], best: str = "latency", transceiver: str = "shannon"
    ) -> List["Connection"]:
        streamed_connections: List[Connection] = []
        for connection in connections:
            if best == "latency":
                path = self.find_best_latency(connection.input_node, connection.output_node)
            elif best == "snr":
                path = self.find_best_snr(connection.input_node, connection.output_node)
            else:
                print("ERROR: best input not recognized. Value:", best)
                continue
            if path:
                assert self.route_space is not None
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = [
                    i for i in range(len(path_occupancy)) if path_occupancy[i] == "free"
                ][0]
                raw_path = path.replace("->", "")
                in_lightpath = Lightpath(raw_path, channel, transceiver=transceiver)
                in_lightpath = self.optimization(in_lightpath)
                out_lightpath = self.propagate(in_lightpath, occupation=True)
                self.calculate_bitrate(out_lightpath)
                if out_lightpath.bitrate == 0.0:
                    for lp in connection.lightpaths:
                        self.update_route_space(lp.path, lp.channel, "free")
                    connection.block_connection()
                else:
                    connection.set_connection(out_lightpath)
                    self.update_route_space(raw_path, out_lightpath.channel, "occupied")
                    if connection.residual_rate_request > 0:
                        self.stream([connection], best, transceiver)
            else:
                connection.block_connection()
            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path: str) -> set:
        path = path.replace("->", "")
        return {path[i] + path[i + 1] for i in range(len(path) - 1)}

    def update_route_space(self, path: str, channel: int, state: str) -> None:
        assert self.route_space is not None
        all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i, line_set in enumerate(all_paths):
            if lines.intersection(line_set):
                states[i] = state
        self.route_space[str(channel)] = states

    def calculate_bitrate(self, lightpath: Lightpath, bert: float = 1e-3) -> float:
        snr = lightpath.snr
        Bn = 12.5e9
        Rs = lightpath.rs
        if lightpath.transceiver.lower() == "fixed-rate":
            snrt = 2 * erfcinv(2 * bert) * (Rs / Bn)
            rb = np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])
        elif lightpath.transceiver.lower() == "flex-rate":
            snrt1 = 2 * erfcinv(2 * bert) ** 2 * (Rs / Bn)
            snrt2 = (14 / 3) * erfcinv(3 / 2 * bert) ** 2 * (Rs / Bn)
            snrt3 = 10 * erfcinv(8 / 3 * bert) ** 2 * (Rs / Bn)
            cond1 = snr < snrt1
            cond2 = snr >= snrt1 and snr < snrt2
            cond3 = snr >= snrt2 and snr < snrt3
            cond4 = snr >= snrt3
            rb = np.piecewise(snr, [cond1, cond2, cond3, cond4], [0, 100, 200, 400])
        elif lightpath.transceiver.lower() == "shannon":
            rb = 2 * Rs * np.log2(1 + snr * (Rs / Bn)) * 1e-9
        lightpath.bitrate = float(rb)
        return float(rb)


class Connection:
    def __init__(self, input_node: str, output_node: str, rate_request: float = 0):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power: Optional[float] = None
        self._latency = 0
        self._snr: List[float] = []
        self._rate_request = float(rate_request)
        self._residual_rate_request = float(rate_request)
        self._lightpaths: List[Lightpath] = []
        self._bitrate: Optional[float] = None

    @property
    def input_node(self) -> str:
        return self._input_node

    @property
    def output_node(self) -> str:
        return self._output_node

    @property
    def rate_request(self) -> float:
        return self._rate_request

    @property
    def residual_rate_request(self) -> float:
        return self._residual_rate_request

    @property
    def bitrate(self) -> Optional[float]:
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate: float) -> None:
        self._bitrate = bitrate

    @property
    def signal_power(self) -> Optional[float]:
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power: float) -> None:
        self._signal_power = signal_power

    @property
    def latency(self) -> float:
        return self._latency

    @latency.setter
    def latency(self, latency: float) -> None:
        self._latency = latency

    @property
    def snr(self) -> List[float]:
        return self._snr

    @snr.setter
    def snr(self, snr: float) -> None:
        self._snr.append(snr)

    @property
    def lightpaths(self) -> List[Lightpath]:
        return self._lightpaths

    @lightpaths.setter
    def lightpaths(self, lightpath: Lightpath) -> None:
        self._lightpaths.append(lightpath)

    def calculate_capacity(self) -> float:
        self.bitrate = sum(lightpath.bitrate for lightpath in self.lightpaths)
        return self.bitrate

    def set_connection(self, lightpath: Lightpath) -> "Connection":
        self.signal_power = lightpath.signal_power
        self.latency = max(self.latency, lightpath.latency)
        self.snr = 10 * np.log10(lightpath.snr)
        self.lightpaths = lightpath
        self._residual_rate_request = self._residual_rate_request - lightpath.bitrate
        return self

    def block_connection(self) -> "Connection":
        self.latency = None
        self.snr = 0
        self.bitrate = 0
        self.clear_lightpaths()
        return self

    def clear_lightpaths(self) -> None:
        self._lightpaths = []


def create_traffic_matrix(nodes: Sequence[str], rate: float, multiplier: float = 5) -> pd.DataFrame:
    base = pd.Series(data=[0.0] * len(nodes), index=nodes)
    df = pd.DataFrame(float(rate * multiplier), index=base.index, columns=base.index, dtype=base.dtype)
    np.fill_diagonal(df.values, base)
    return df


def plot3Dbars(matrix: pd.DataFrame) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x_data, y_data = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = matrix.values.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    plt.show()


def create_connections(node_labels: Sequence[str], traffic: pd.DataFrame) -> List[Connection]:
    node_pairs = list(filter(lambda x: x[0] != x[1], list(it.product(node_labels, node_labels))))
    shuffle(node_pairs)
    connections: List[Connection] = []
    for node_pair in node_pairs:
        connections.append(
            Connection(node_pair[0], node_pair[-1], float(traffic.loc[node_pair[0], node_pair[-1]]))
        )
    return connections


def run_monte_carlo(
    topology_path: str,
    channels: int,
    iterations: int,
    upgrade_line: str,
    traffic_rate: float,
    traffic_multiplier: float,
    best: str,
    transceiver: str,
) -> Tuple[List[List[Connection]], List[Dict[str, Line]]]:
    stream_conn_list: List[List[Connection]] = []
    lines_state_list: List[Dict[str, Line]] = []
    for i in range(iterations):
        print(f"Monte Carlo Realization #{i + 1}")
        network = Network(topology_path, nch=channels, upgrade_line=upgrade_line)
        network.connect()
        node_labels = list(network.nodes.keys())
        traffic = create_traffic_matrix(node_labels, traffic_rate, multiplier=traffic_multiplier)
        connections = create_connections(node_labels, traffic)
        streamed_connections = network.stream(connections, best=best, transceiver=transceiver)
        stream_conn_list.append(streamed_connections)
        lines_state_list.append(network.lines)
    return stream_conn_list, lines_state_list


def summarize_results(stream_conn_list: List[List[Connection]], lines_state_list: List[Dict[str, Line]]) -> None:
    snr_conns: List[List[float]] = []
    rbl_conns: List[List[float]] = []
    rbc_conns: List[List[float]] = []
    for streamed_conn in stream_conn_list:
        snrs: List[float] = []
        rbl: List[float] = []
        for connection in streamed_conn:
            snrs.extend(connection.snr)
            for lightpath in connection.lightpaths:
                rbl.append(lightpath.bitrate)
        rbc = [connection.calculate_capacity() for connection in streamed_conn]
        snr_conns.append(snrs)
        rbl_conns.append(rbl)
        rbc_conns.append(rbc)

    lines_labels = list(lines_state_list[0].keys())
    congestions = {label: [] for label in lines_labels}
    for line_state in lines_state_list:
        for line_label, line in line_state.items():
            cong = line.state.count("occupied") / len(line.state)
            congestions[line_label].append(cong)
    avg_congestion = {label: float(np.mean(cong)) for label, cong in congestions.items()}
    plot_congestion(avg_congestion)

    traffic_list = [np.sum(rbl_list) for rbl_list in rbl_conns]
    avg_rbl_list = [np.mean(rbl_list) for rbl_list in rbl_conns]
    avg_snr_list = [
        np.mean(list(filter(lambda x: x != 0, snr_list))) for snr_list in snr_conns
    ]

    print("\n")
    print(f"Line to upgrade: {max(avg_congestion, key=avg_congestion.get)}")
    print(f"Avg Total Traffic: {np.mean(traffic_list) * 1e-3:.2f} Tbps")
    print(f"Avg Lightpath Bitrate: {np.mean(avg_rbl_list):.2f} Gbps")
    print(f"Avg Lightpath SNR: {np.mean(avg_snr_list):.2f} dB")

    for snrs in snr_conns:
        plt.hist(snrs, bins=10)
        plt.title("SNR Distribution [dB]")


def plot_congestion(avg_congestion: Dict[str, float]) -> None:
    plt.bar(range(len(avg_congestion)), list(avg_congestion.values()), align="center")
    plt.xticks(range(len(avg_congestion)), list(avg_congestion.keys()))
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Network simulator for optical lightpaths.")
    parser.add_argument("--topology", default="nodes_9.json", help="Path to topology JSON file.")
    parser.add_argument("--channels", type=int, default=10, help="Number of channels per line.")
    parser.add_argument("--iterations", type=int, default=2, help="Monte Carlo iterations.")
    parser.add_argument("--upgrade-line", default="DB", help="Line label to upgrade (noise figure reduction).")
    parser.add_argument("--traffic-rate", type=float, default=600, help="Base traffic rate.")
    parser.add_argument(
        "--traffic-multiplier",
        type=float,
        default=5,
        help="Traffic multiplier for traffic matrix.",
    )
    parser.add_argument(
        "--best",
        choices=("latency", "snr"),
        default="snr",
        help="Optimization target for routing.",
    )
    parser.add_argument(
        "--transceiver",
        choices=("shannon", "fixed-rate", "flex-rate"),
        default="shannon",
        help="Transceiver mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream_conn_list, lines_state_list = run_monte_carlo(
        topology_path=args.topology,
        channels=args.channels,
        iterations=args.iterations,
        upgrade_line=args.upgrade_line,
        traffic_rate=args.traffic_rate,
        traffic_multiplier=args.traffic_multiplier,
        best=args.best,
        transceiver=args.transceiver,
    )
    summarize_results(stream_conn_list, lines_state_list)


if __name__ == "__main__":
    main()
