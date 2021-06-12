import json
from builtins import list
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import Planck, c, pi
from scipy.special import erfcinv


class Lightpath(object):
    def __init__(self, path, power=None, channel=0, rs=32e9, df=50e9, transceiver='flex-rate'):
        self._path = path
        self._signal_power = power
        self._channel = channel
        self._rs = rs
        self._df = df
        self._latency = 0
        self._noise_power = 0
        self._snr = None
        self._optimized_powers = {}
        self._transceiver = transceiver
        self._bitrate = None

    @property
    def tranceiver(self):
        return  self._transceiver

    @tranceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    @property
    def optimized_powers(self):
        return self._optimized_powers

    @optimized_powers.setter
    def optimized_powers(self, optimized_powers):
        self._optimized_powers = optimized_powers

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @property
    def channel(self):
        return self._channel

    @property
    def rs(self):
        return self._rs

    @property
    def df(self):
        return self._df

    @path.setter
    def path(self, path):
        self._path = path

    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency


    def add_noise(self, noise):
        self._noise_power += noise

    def add_latency(self, latency):
        self._latency += latency

    def next(self):
        self._path = self._path[1:]

    def update_snr(self, snr):
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    # lab8 ftn()
    def optimize(self, lightpath):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]

            ase = line.ase_generation()
            eta = line.nli_generation(1, lightpath.rs, lightpath.df)
            p_opt = (ase / (2 * eta)) ** (1/3)
            lightpath.optimized_powers.update({line_label: p_opt})

            lightpath.next()
            node = line.successive[lightpath.path[0]]
            lightpath = node.optimize(lightpath)
        return lightpath


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length']*1e3
        self._amplifiers=int(np.ceil(self._length/80e3))
        self._span_length = self._length/self._amplifiers
        self._noise_figure=5
        self._state = ['free'] * 10
        self.successive = {}

        # Physical parameters of the fiber
        self._alpha=4.6e-5
        self._beta=21.27e-27
        self._gama=1.27e-3

        #Set Gain to transparency
        self._gain=self.transparency()

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def gain(self):
        return self._gain
    @gain.setter
    def gain(self,gain):
        self._gain=gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def span_lenght(self):
        return self._span_length

    @property
    def amplifiers(self):
        return self._amplifier

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gama(self):
        return self._gama

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        state = [s.lower().strip() for s in state]
        if set(state).issubset(set(['free', 'occupied'])):
            self._state = state
        else:
            print("Error: line state not recognized. Value: ", set(state) - set(['free', 'occupied']))

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def latency_generation(self):
        return self._length / (c * 2 / 3)

    def noise_generation(self, lightpath):
        # print(type(lightpath))
        noise = self.ase_generation()+self.nli_generation(lightpath.signal_power,lightpath.rs,lightpath.df)
        return noise

    def ase_generation(self):
        gain_lin=10**(self._gain/10)
        noise_figure_lin=10**(self._noise_figure/10)
        N=self._amplifiers
        f=193.4e12
        Bn=12.5e9
        h=Planck
        ase_noise=N*h*f*Bn*noise_figure_lin*(gain_lin-1)
        return ase_noise

    def nli_generation(self,signal_power,rs,df):
        Nch=10
        Pch=signal_power
        Bn=12.5e9
        loss=np.exp(-self._alpha*self._span_length)
        gain_lin=10**(self._gain/10)
        N_spans=self._amplifiers
        eta=16/(27*pi)*\
            np.log(pi**2*self._beta*rs**2*\
            Nch**(2*rs/df)/(2*self._alpha))*\
            self._gama**2/(4*self._alpha*self._beta*rs**3)
        nli_noise = N_spans*(Pch**3*loss*gain_lin*eta*Bn)
        return nli_noise

    def transparency(self):
        gain = 10*np.log10(np.exp(self._alpha*self._span_length))
        return gain




    def propagate(self, lightpath, occupation=False):
        # Update Latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update Noise
        #signal_power = lightpath._signal_power
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)

        #update SNR
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)

        # Update Line State
        if occupation:
            channel = lightpath._channel
            new_state = self._state.copy()
            new_state[channel] = 'occupied'
            self._state = new_state

        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)
        return lightpath


class Connection(object):

    def __init__(self, input_node, output_node):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = None
        self._latency = 0
        self._snr = 0
        self._bitrate  = None

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    def calculate_capacity(self):
        return self._bitrate

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def latency(self):
        return self._latency

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, power):
        self._signal_power = power

    @property
    def snr(self):
        return self._snr

    @latency.setter
    def latency(self, latency):
        self._latency = latency
    @snr.setter
    def snr(self, snr):
        self._snr = snr


class Network(object):

    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}

        self._connected = False
        self._weighted_paths = None

        self._route_space = None

        # Creating Node Instance
        for node_label in node_json:
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # Creating Line Instance
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                length = np.sqrt(np.sum(np.square(node_position - connected_node_position)))
                line_dict['length'] = length
                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def route_space(self):
        return self._route_space

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def set_weighted_paths(self):
        if not self._connected:
            self.connect()
        node_labels = network.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)
        columns = ['path', 'latency', 'noise', 'snr']
        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in pairs:
            for path in network.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # propagation
                ligthpath = Lightpath(path)
                ligthpath = self.optimization(ligthpath)
                ligthpath = self.propagate(ligthpath, occupation=False)

                latencies.append(ligthpath.latency)
                noises.append(ligthpath.noise_power)
                snrs.append(10 * np.log10(ligthpath.snr))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df
        route_space = pd.DataFrame()
        route_space['path'] = paths
        for i in range(10):
            route_space[str(i)] = ['free'] * len(paths)
        self._route_space = route_space

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys()
                       if ((key != label2) & (key != label1))]
        cross_lines = self.lines.keys()
        inner_paths = {}
        inner_paths['0'] = label1
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node
                                            for cross_node in cross_nodes
                                            if ((cross_node not in inner_path) & (
                                inner_path[-1] + cross_node in cross_lines))]

        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        self._connected = True
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node_label in node.connected_nodes:
                line_label = node_label + connected_node_label
                line = lines_dict[line_label]
                line.successive[connected_node_label] = nodes_dict[connected_node_label]
                node.successive[line_label] = lines_dict[line_label]

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_lightpath = start_node.propagate(lightpath, occupation)
        return propagated_lightpath

    def stream(self, connections, best='latency', transceiver='flex-rate'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            self.set_weighted_paths()
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('Error: best input not recognized. Value: ', best)
                continue
            if path:
                path_occupancy = self._route_space.loc[self._route_space.path == path].T.values[1:]
                # print(self._route_space)

                channel = [i for i in range(len(path_occupancy))   # Uses First Free Channel
                           if path_occupancy[i] == 'free'][0]
                path = path.replace('->', '')

                in_lightpath = Lightpath(path, channel, transceiver=transceiver)
                in_lightpath = self.optimization(in_lightpath)
                out_lightpath = self.propagate(in_lightpath, True)

                connection.signal_power = out_lightpath.signal_power
                connection.latency = out_lightpath.latency
                connection.snr = 10 * np.log10(out_lightpath.snr)
                connection.bitrate = self.calculate_bitrate(out_lightpath)
                if not connection.bitrate:
                    connection.latency = None
                    connection.snr = 0
                    connection.bitrate = 0
                self.update_route_space(path, channel)
            else:
                connection._latency = None
                connection._snr = 0
                connection.birate = 0

            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    def available_paths(self, input_node, output_node):
        if self._weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths = [path for path in self._weighted_paths.path.values
                     if ((path[0] == input_node) & (path[-1] == output_node))]
        available_paths = []
        for path in all_paths:
            path_occupacy = self._route_space.loc[self._route_space.path == path].T.values[1:]
            if 'free' in path_occupacy:
                available_paths.append(path)
        return available_paths

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self._weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self._weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None
        return best_path

    def update_route_space(self, path, channel):
        all_paths = [self.path_to_line_set(p)
                     for p in self._route_space.path.values]
        states = self._route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = 'occupied'
        self._route_space[str(channel)] = states


    def optimization(self,lightpath):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)
        optimized_lightpath.path = path
        return optimized_lightpath

    # lab8 ftn()
    def calculate_bitrate(self, lightpath, bert=1e-3):
        snr = lightpath.snr
        Bn = 12.5e9
        Rs = lightpath.rs

        if lightpath.transceiver.lower() == 'fixed-rate':
            snrt = 2 * erfcinv(2 * bert) * (Rs / Bn)
            rb = np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])
        elif lightpath.transceiver.lower() == 'flex-rate':
            snrt1 = 2 * erfcinv(2 * bert) ** 2 * (Rs / Bn)
            snrt2 = (14/3) * erfcinv(3/2 * bert) ** 2 * (Rs / Bn)
            snrt3 = 10 * erfcinv(8/3 * bert) ** 2 * (Rs / Bn)

            cond1 = (snr < snrt1)
            cond2 = (snr >= snrt1 and snr < snrt2)
            cond3 = (snr >= snrt2 and snr < snrt3)
            cond4 = (snr >= snrt3)
            rb = np.piecewise(snr, \
                              [cond1, cond2, cond3, cond4], [0, 100, 200, 400])
        elif lightpath.transceiver.lower() == 'shannon':
            rb = 2 * Rs * np.log2(1 + snr * (Rs / Bn)) * 1e-9

        lightpath.bitrate = float(rb)
        return float(rb)

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0 + 20, y0 + 20, node_label)

            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('Network')
        plt.show()


network = Network('nodes.json')
network.connect()
node_labels = list(network.nodes.keys())
connections = []
for i in range(1000):
    shuffle(node_labels)
    connection = Connection(node_labels[0], node_labels[-1])
    connections.append(connection)

streamed_connections = network.stream(connections, best='snr', transceiver='shannon')
snrs = [connection.snr for connection in streamed_connections]
plt.hist(snrs, bins=10)
plt.title('SNR Distribution')
plt.show()

rbs = [connection.calculate_capacity() \
       for connection in streamed_connections]
plt.hist(rbs, bins=10)
plt.title('Bitrate Distribution [Gbps]')
plt.show()

print('Total Capacity: {:.2f} Tbps'.format(np.sum(rbs) * 1e-3))
print('Avg Capacity: {:.2f} Gbps'.format(np.mean(rbs)))
