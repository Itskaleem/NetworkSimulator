import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c
from random import shuffle

class SignalInformation(object):
    def __init__(self,power, path):
        self._path=path
        self._signal_power=power
        self._latency=0
        self._noise_power=0

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path
    @path.setter
    def path(self,path):
        self._path=path

    @property
    def noise_power(self):
        return self._noise_power
    @noise_power.setter
    def noise_power(self,noise):
        self._noise_power=noise

    @property
    def latency(self):
        return self._latency
    @latency.setter
    def latency(self,latency):
        self._latency=latency

    def add_noise(self,noise):
        self._noise_power+=noise

    def add_latency(self,latency):
        self._latency+=latency

    def next(self):
        self._path=self._path[1:]

class Lightpath(SignalInformation):
    def __init__(self,power,path,channel):
        self._path = path
        self._signal_power = power
        self._channel=channel
        self._latency = 0
        self._noise_power = 0

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @property
    def channel(self):
        return self._channel

    @path.setter
    def path(self, path):
        self._path = path

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

class Node(object):
    def __init__(self,node_dict):
        self._label=node_dict['label']
        self._position=node_dict['position']
        self._connected_nodes=node_dict['connected_nodes']
        self._successive={}

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

    def propagate(self,lightpath,occupation=False):
        path=lightpath.path
        if len(path)>1:
            line_label=path[:2]
            line=self.successive[line_label]
            lightpath.next()
            lightpath=line.propagate(lightpath,occupation)
        return lightpath

class Line(object):
    def __init__(self,line_dict):
        self._label=line_dict['label']
        self._length=line_dict['length']
        self._state=['free']*10
        self.successive={}

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @property
    def state(self):
        return self._state
    @state.setter
    def state(self,state):
        state=[s.lower().strip() for s in state]
        if set(state).issubset(set(['free','occupied'])):
            self._state=state
        else:
            print("Error: line state not recognized. Value: ",set(state)-set(['free','occupied']))



    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self,successive):
        self._successive=successive

    def latency_generation(self):
        return self._length/(c*2/3)

    def noise_generation(self,signal_power):
        return signal_power/(2*self._length)

    def propagate(self,lightpath,occupation=False):
        #Update Latency
        latency=self.latency_generation()
        lightpath.add_latency(latency)
        #Update Noise
        signal_power=lightpath._signal_power
        noise=self.noise_generation(signal_power)
        lightpath.add_noise(noise)
        #Update Line State
        if occupation:
            channel=lightpath._channel
            new_state=self._state.copy()
            new_state[channel]='occupied'
            self._state=new_state

        node=self.successive[lightpath.path[0]]
        lightpath=node.propagate(lightpath,occupation)
        return lightpath

class Connection(object):

    def __init__(self,input_node,output_node,signal_power):
        self._input_node=input_node
        self._output_node=output_node
        self._signal_power=signal_power
        self._latency=0
        self._snr=0

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

    @property
    def snr(self):
        return self._snr

    def latency_seter(self,latency):
        self._latency=latency

    def snr_setter(self,snr):
        self._snr=snr

class Network(object):

    def __init__(self,json_path):
        node_json=json.load(open(json_path,'r'))
        self._nodes={}
        self._lines={}

        self._connected=False
        self._weighted_paths=None

        self._route_space=None


        # Creating Node Instance
        for node_label in node_json:
            node_dict=node_json[node_label]
            node_dict['label']=node_label
            node=Node(node_dict)
            self._nodes[node_label]=node

        # Creating Line Instance
            for connected_node_label in node_dict['connected_nodes']:
                line_dict={}
                line_label=node_label+connected_node_label
                line_dict['label']=line_label
                node_position=np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                length=np.sqrt(np.sum(np.square(node_position-connected_node_position)))
                line_dict['length']=length
                line=Line(line_dict)
                self._lines[line_label]=line


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

    def set_weighted_paths(self,signal_power):
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
                signal_information = SignalInformation(signal_power, path)
                signal_information = self.propagate(signal_information,occupation=False)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths=df
        route_space=pd.DataFrame()
        route_space['path']=paths
        for i in range(10):
            route_space[str(i)]=['free']*len(paths)
        self._route_space=route_space

    def find_paths(self,label1,label2):
        cross_nodes=[key for key in self.nodes.keys()
                            if((key!=label2)&(key!=label1))]
        cross_lines=self.lines.keys()
        inner_paths={}
        inner_paths['0']=label1
        for i in range(len(cross_nodes)+1):
            inner_paths[str(i+1)]=[]
            for inner_path in inner_paths[str(i)]:
                inner_paths[str(i+1)] += [inner_path+cross_node
                                         for cross_node in cross_nodes
                                         if((cross_node not in inner_path) & (inner_path[-1]+cross_node in cross_lines))]

        paths=[]
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1]+label2 in cross_lines:
                    paths.append(path+label2)
        return paths


    def connect(self):
        self._connected=True
        nodes_dict=self.nodes
        lines_dict=self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node_label in node.connected_nodes:
                line_label=node_label+connected_node_label
                line=lines_dict[line_label]
                line.successive[connected_node_label]=nodes_dict[connected_node_label]
                node.successive[line_label]=lines_dict[line_label]

    def propagate(self,lightpath,occupation=False):
        path=lightpath.path
        start_node=self.nodes[path[0]]
        propagated_lightpath=start_node.propagate(lightpath,occupation)
        return propagated_lightpath


    def stream(self,connections,best='latency'):
        streamed_connections=[]
        for connection in connections:
            input_node=connection.input_node
            output_node=connection.output_node
            signal_power=connection.signal_power
            self.set_weighted_paths(1)
            if best == 'latency':
                path=self.find_best_latency(input_node,output_node)
            elif best == 'snr':
                path=self.find_best_snr(input_node,output_node)
            else:
                print('Error: best input not recognized. Value: ',best)
                continue
            if path:
                path_occpacy=self._route_space.loc[self._route_space.path==path].T.values[1:]
               # print(self._route_space)

                channel=[i for i in range(len(path_occpacy))
                         if path_occpacy[i]=='free'][0]
                path=path.replace('->','')
                in_lightpath=Lightpath(signal_power,path,channel)
                out_lightpath=self.propagate(in_lightpath,True)
                connection._latency=out_lightpath.latency
                noise_power=out_lightpath.noise_power
                connection._snr=10*np.log10(signal_power/noise_power)
                self.update_route_space(path,channel)
            else:
                connection._latency=None
                connection._snr=0
            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path=path.replace('->','')
        return set([path[i]+path[i+1] for i in range(len(path)-1)])

    def available_paths(self,input_node,output_node):
        if self._weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths=[path for path in self._weighted_paths.path.values
                   if((path[0]==input_node)&(path[-1]==output_node))]
        available_paths=[]
        for path in all_paths:
            path_occupacy=self._route_space.loc[self._route_space.path==path].T.values[1:]
            if 'free' in path_occupacy:
                available_paths.append(path)
        return available_paths
    
    def find_best_snr(self,input_node,output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df=self.weighted_paths.loc[self._weighted_paths.path.isin(available_paths)]
            best_snr=np.max(inout_df.snr.values)
            best_path=inout_df.loc[inout_df.snr==best_snr].path.values[0]
        else:
            best_path=None
        return best_path

    def find_best_latency(self,input_node,output_node):
        available_paths = self.available_paths(input_node, output_node)
        if available_paths:
            inout_df=self.weighted_paths.loc[self._weighted_paths.path.isin(available_paths)]
            best_latency=np.min(inout_df.latency.values)
            best_path=inout_df.loc[inout_df.latency==best_latency].path.values[0]
        else:
            best_path=None
        return best_path

    def update_route_space(self,path,channel):
        all_paths=[self.path_to_line_set(p)
                   for p in self._route_space.path.values]
        states=self._route_space[str(channel)]
        lines=self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set=all_paths[i]
            if lines.intersection(line_set):
                states[i]='occupied'
        self._route_space[str(channel)]=states

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
node_labels=list(network.nodes.keys())
connections=[]
for i in range(1000):
    shuffle(node_labels)
    connection=Connection(node_labels[0],node_labels[-1],1)
    connections.append(connection)


streamed_connections = network.stream(connections,best='snr')
snrs=[connection.snr for connection in streamed_connections]
plt.hist(snrs,bins=10)
plt.title('SNR Distribution')
plt.show()
