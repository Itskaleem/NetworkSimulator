from random import shuffle
import itertools as it
from scipy.constants import Planck, c, pi
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import erfcinv
import json
import numpy as np



class Lightpath(object):
    def __init__(self, path: str, channel=0, rs=32e9, df=50e9, transceiver ='shannon'):
        self._signal_power = None
        self._path = path
        self._channel = channel
        self._rs = rs
        self._df = df
        self._noise_power = 0
        self._snr = None
        self._latency = 0
        self._optimized_powers = {}
        self._transceiver = transceiver
        self._bitrate = None

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    @property
    def signal_power(self):
        return self._signal_power

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

    def update_snr(self, snr):
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def rs(self):
        return self._rs

    @property
    def df(self):
        return self._df

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def channel(self):
        return self._channel

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
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self.path = self.path[1:]


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

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    def optimize(self, lightpath):
        path = lightpath.path

        if len(path) > 1:
           line_label = path[:2]
           line = self.successive[line_label]
           ase = line.ase_generation()
           eta = line.nli_generation(1, lightpath.rs, lightpath.df)
           p_opt = (ase / (2 * eta)) ** (1 / 3)  # calculate optimum signal power
           lightpath.optimized_powers.update({line_label: p_opt})
           lightpath.next()
           node = line.successive[lightpath.path[0]]
           lightpath = node.optimize(lightpath)
        return lightpath


class Connection(object):
    def __init__(self, input_node, output_node, rate_request =0):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = None
        self._latency = 0
        self._snr = []
        self._rate_request = float(rate_request)
        self._residual_rate_request = float(rate_request)
        self._lightpaths = []
        self._bitrate = None # removed in the code

    @property
    def input_node(self):
        return self._input_node

    @property
    def rate_request(self):
        return self._rate_request
    @property
    def residual_rate_request(self):
        return self._residual_rate_request
    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    def calculate_capacity(self):
        self.bitrate = sum([lightpath.bitrate for lightpath in self.lightpaths])
        return self.bitrate

    def set_connection(self, lightpath):
        self.signal_power = lightpath.signal_power
        self.latency = max(self.latency, lightpath.latency)
        self.snr = 10 * np.log10(lightpath.snr)
        self.lightpaths = lightpath
        self._residual_rate_request = self._residual_rate_request - lightpath.bitrate
        return self

    def block_connection(self):
        self.latency = None
        self.snr = 0
        self.bitrate = 0
        self.clear_lightpaths()
        return self


    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    @property
    def lightpaths(self):
        return self._lightpaths

    @lightpaths.setter
    def lightpaths(self, lightpath):
        self._lightpaths.append(lightpath)

    @snr.setter
    def snr(self, snr):
        self._snr.append(snr)

    def clear_lightpaths(self):
        self._lightpaths = []


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._length = line_dict['length'] * 1e3
        self._amplifiers = int(np.ceil(self._length / 80e3))
        self._span_length = self._length / self._amplifiers
        # Set Gain to transparency
        self._noise_figure = 7 # increased noise figure to see more results such as different rates for the lightpaths bcz the rate is lower
        # Physical Parameters of the Fiber
        self._alpha = 4.6e-5
        self._beta = 21.27e-27
        self._gamma = 1.27e-3
        self._state = ['free'] * 10
        self._successive = {}
        self._gain = self.transparency()

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def state(self):
        return self._state

    @property
    def amplifiers(self):
        return self._amplifiers

    @property
    def span_length(self):
        return self._span_length

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @state.setter
    def state(self, state):
        state = [s.lower().strip() for s in state]
        if set(state).issubset(set(['free', 'occupied'])):
            self._state = state
        else:
            print('ERROR: line state  not recognized.Value:', set(state) - set(['free', 'occupied']))

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def transparency(self):
        gain = 10 * np.log10(np.exp(self.alpha * self.span_length))

        return gain

    def latency_generation(self):
        latency = self.length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, lightpath.rs, lightpath.df)

        return noise

    def ase_generation(self):
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        N = self._amplifiers
        f = 193.4e12
        h = Planck
        Bn = 12.5e9
        ase_noise = N * h * f * Bn * noise_figure_lin * (gain_lin - 1)
        return ase_noise

    def nli_generation(self, signal_power, Rs, df):
        Nch = 10
        Pch = signal_power
        Bn = 12.5e9
        loss = np.exp(- self.alpha * self.span_length)  # modified this line
        gain_lin = 10 ** (self.gain / 10)
        N_spans = self.amplifiers
        eta = 16 / (27 * pi) * \
              np.log(pi ** 2 * self.beta * Rs ** 2 * \
                     Nch ** (2 * Rs / df) / (2 * self.alpha)) * \
              self.gamma ** 2 / (4 * self.alpha * self.beta * Rs ** 3)
        nli_noise = N_spans * (Pch ** 3 * loss * gain_lin * eta * Bn)
        return nli_noise

    def propagate(self, lightpath, occupation=False):
        # Update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)
        # Update noise
        signal_power = lightpath.signal_power
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)
        # Update SNR
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)
        # Update line state
        if occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state
        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)
        return lightpath




class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None
        self._route_space = None
        for node_label in node_json:
            # Create the node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            # Create the line instances
            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label + connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['length'] = np.sqrt(np.sum((node_position - connected_node_position) ** 2))

                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def connected(self):
        return self._connected

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

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

    def find_paths(self, label1, label2):
        cross_nodes = [key for key in self.nodes.keys() if ((key != label1) & (key != label2))]
        cross_lines = self.lines.keys()
        inner_paths = {}
        inner_paths['0'] = label1
        for i in range(len(cross_nodes) + 1):
            inner_paths[str(i + 1)] = []
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i + 1)] += [inner_path + cross_node
                                            for cross_node in cross_nodes
                                            if ((inner_path[-1] + cross_node in cross_lines) &
                                                (cross_node not in inner_path))]
        paths = []
        for i in range(len(cross_nodes) + 1):
            for path in inner_paths[str(i)]:
                if path[-1] + label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label + connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
        self._connected = True

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_lightpath = start_node.propagate(lightpath, occupation)
        return propagated_lightpath

    def optimization(self, lightpath):
        # sets the lightpath power to the optimal \
        # power calculated at each line ( node beginning the line)
        path = lightpath . path
        start_node = self . nodes[path[0]]
        optimized_lightpath = start_node . optimize(lightpath)
        optimized_lightpath . path = path
        return optimized_lightpath


    def set_weighted_paths(self):  # in the pdf he removed the signal power
        if not self.connected:
            self.connect()
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)

        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string += node + '->'
                paths.append(path_string[:-2])

                # Propagation
                lightpath = Lightpath(path) # need to pass more
                lightpath = self.optimization(lightpath)
                lightpath = self.propagate(lightpath, occupation=False)

                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(
                    10 * np.log10(lightpath.signal_power / lightpath.noise_power)
                )
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

    def available_paths(self, input_node, output_node):  # path liberi per quella copia di nodi ma su tutti i canali
        if self.weighted_paths is None:
            self.set_weighted_paths()
        all_paths = [path for path in self.weighted_paths.path.values
                     if ((path[0] == input_node) and (path[-1] == output_node))]
        available_paths = []
        for path in all_paths:
            path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[
                             1:]  # route space è lista di path con una entry per ogni lambda
            if 'free' in path_occupancy:  # se i canali sono tuttio occupati esclude il path
                available_paths.append(path)
        return available_paths

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None
        return best_path

    def stream(self, connections, best='latency',  transceiver ='shannon'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('ERROR: best input not recognized. Value:', best)
                continue
            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]  # prende il primo canale libero
                path = path.replace('->', '')
                in_lightpath = Lightpath (path , channel , transceiver = transceiver )
                in_lightpath = self . optimization ( in_lightpath )
                out_lightpath = self . propagate ( in_lightpath , occupation = True )
                self.calculate_bitrate(out_lightpath)
                if out_lightpath.bitrate == 0.0:
                    #[self.update_route_space(lp.path, lp.channel,'free') for lp in connection.lightpaths]
                    connection.block_connection()
                else:
                    connection.set_connection(out_lightpath)
                    self.update_route_space(path, out_lightpath.channel,'occupied') #here
                    if connection.residual_rate_request > 0:
                        self.stream([connection], best, transceiver) #removed type
            else:
                #[self.update_route_space(lp.path, lp.channel, 'free') for lp in connection.lightpaths]
                connection.block_connection()
            streamed_connections.append(connection)
        return streamed_connections


    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i + 1] for i in range(len(path) - 1)])

    def update_route_space(self, path, channel,state):
        all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = state
        self.route_space[str(channel)] = states

    def calculate_bitrate(self, lightpath, bert=1e-3):
        snr = lightpath.snr
        Bn = 12.5e9
        Rs = lightpath.rs
        if lightpath.transceiver.lower() == 'fixed-rate':
           snrt = 2 * erfcinv(2 * bert) * (Rs / Bn)
           rb = np.piecewise(snr, [snr < snrt, snr >= snrt], [0, 100])
        elif lightpath.transceiver.lower() == 'flex-rate':
           snrt1 = 2 * erfcinv(2 * bert) ** 2 * (Rs / Bn)
           snrt2 = (14 / 3) * erfcinv(3 / 2 * bert) ** 2 * (Rs / Bn)
           snrt3 = 10 * erfcinv(8 / 3 * bert) ** 2 * (Rs / Bn)
           cond1 = (snr < snrt1)
           cond2 = (snr >= snrt1 and snr < snrt2)
           cond3 = (snr >= snrt2 and snr < snrt3)
           cond4 = (snr >= snrt3)
           rb = np.piecewise(snr, \
                      [cond1, cond2, cond3, cond4], [0, 100, 200, 400])
        elif lightpath.transceiver.lower() =='shannon':
             rb = 2 * Rs * np.log2(1 + snr * (Rs / Bn)) * 1e-9
        lightpath.bitrate = float(rb)  # set bitrate in lightpath
        return float(rb)





def create_traffic_matrix(nodes, rate):
    s = pd.Series(data=[0.0] * len(nodes), index=nodes)
    df = pd.DataFrame(float(rate), index=s.index, columns=s.index, dtype=s.dtype)
    np.fill_diagonal(df.values, s)
    return df


def plot3Dbars(t):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_data, y_data = np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = t.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    plt.show()


def main():
    network = Network('nodes_2.json')  # creates nodes and line objects
    network.connect()  # connects the net by setting the line successive attribute with the node object at the end of the line
    network.draw()
    node_labels = list(network.nodes.keys())
    T = create_traffic_matrix(node_labels, 3000) # 200 connection reqs
    t = T.values
    connections = []
    node_pairs = list(filter(lambda x: x[0] != x[1], list(it.product(node_labels, node_labels))))
    shuffle(node_pairs)
    for node_pair in node_pairs:
        connection = Connection(node_pair[0], node_pair[-1], float(T.loc[node_pair[0], node_pair[-1]]))
        connections.append(connection)  # list of connection objects
    streamed_connections = network.stream(connections, best='snr', transceiver='shannon')
    snrs = []
    [snrs.extend(connection.snr) for connection in streamed_connections]
    rbl = []
    for connection in streamed_connections:
        for lightpath in connection.lightpaths:
            rbl.append(lightpath.bitrate)
    # Plot
    plt.hist(snrs, bins=10)
    plt.title('SNR Distribution[dB]')
    plt.show()
    rbc = [connection.calculate_capacity() for connection in streamed_connections]
    plt.hist(rbc, bins=10)
    plt.title('Connection Capacity Distribution[Gbps]')
    plt.show()
    plt.hist(rbl, bins=10)
    plt.title('Lightpaths Capacity Distribution[Gbps]')
    plt.show()
    s = pd.Series(data=[0.0] * len(node_labels), index=node_labels)
    df = pd.DataFrame(0.0, index=s.index, columns=s.index, dtype=s.dtype)
    print(df)
    for connection in streamed_connections:
        df.loc[connection.input_node, connection.output_node] = connection.bitrate
    print(df)
    plot3Dbars(t)
    plot3Dbars(df.values)
    print('Avg SNR: {:.2f} dB', format(np.mean(list(filter(lambda x: x != 0, snrs)))))
    print('Total Capacity Connections: {:.2f} Tbps ', format(np.sum(rbc) * 1e-3))
    print('Total Capacity Lightpaths: {:.2f} Tbps', format(np.sum(rbl) * 1e-3))
    print('AvgCapacity: {:.2 f} Gbps', format(np.mean(rbc)))


main()