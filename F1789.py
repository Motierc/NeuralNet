import random
import os
import math

NETWORK_STORAGE_LOCATION = r"XXX\XXXX\XXX"

class BaseNeuron: # Used for process node and for input node
  def __init__(self, name: str = "", activation: float = 0) -> None:
    self.name = name
    self.activation = activation

  def set_activation(self, activation: float) -> None:
    self.activation = activation

  def get_activation(self) -> float: 
    return self.activation
  
  def show(self) -> tuple[str, float]:
    return self.name, self.activation
  
  def get_name(self) -> str: 
    return self.name

class ProcessNeuron(BaseNeuron): # main nodes in the neural network
  def __init__(self, name: str) -> None:
    self.attached_nodes: list[BaseNeuron]
    self.attached_nodes = []
    self.weights: list[float]
    self.biases: list[float]
    self.activation: float
    self.activation = 0
    self.name = name

  def attach(self, nodes: list[BaseNeuron]) -> None:
    self.attached_nodes = nodes
    self.weights = [random.random()-0.5 for _ in range(len(nodes))]
    self.biases = [random.random()-0.5 for _ in range(len(nodes))]

  def add_node(self, nodes: list) -> None:
    for node in nodes:
        self.attached_nodes.append(node)
        self.weights.append(random.randrange(1, 100) - 0.5 for _ in range(len(nodes)))
        self.biases.append(random.randrange(1 ,100) - 0.5 for _ in range(len(nodes)))

  def calculate_activation(self) -> None:
    self.activation = 0
    node_activations = [node.get_activation() for node in self.attached_nodes]
    for node_activation, node_weight, node_bias in zip(node_activations, self.weights, self.biases):
      self.activation += (node_activation * node_weight) + node_bias

  def get_activation(self) -> float:
    self.calculate_activation()
    return self.activation
  
  def show(self, display: bool = True) -> list[str, list[str], list[float], list[float]]:
    nod_names = [node.get_name() for node in self.attached_nodes]
    if display:
      print(f'"{self.name}" (with activation {self.activation}) is connected to {len(nod_names)} node(s):')
      for i in range(len(nod_names)):
        print(f'  {i+1}) "{nod_names[i]}" with a weight of {self.weights[i]} and bias of {self.biases[i]}')
    return [self.name, nod_names, self.weights, self.biases]

  def get_data(self):
    return [self.weights, self.biases]

  def set_data(self, nodes: list[BaseNeuron], weights: list[float], biases: list[float]):
    self.attached_nodes = nodes
    self.weights = weights
    self.biases = biases

class Network:
  def __init__(self, name: str, network_structure: list[int]) -> None:
    self.name = name
    self.network: list[list[BaseNeuron]]
    self.network = [[] for _ in range(len(network_structure))]
    self.network[0] = [BaseNeuron(f"I.{i}") for i in range(network_structure[0])]

    for i, layer_size in enumerate(network_structure[1:]): # All the layers except for input layer
      self.network[i+1] = [ProcessNeuron(f"P.{i}.{j}") for j in range(layer_size)]
      for neuron in self.network[i+1]:
        neuron.attach(self.network[i])

  def set_inputs(self, inputs: list[float]) -> None:
    for neuron, input in zip(self.network[0], inputs):
      neuron.set_activation(input)

  def get_outputs(self) -> list[float]:
    ret_list = [out_node.get_activation() for out_node in self.network[-1]]
    return ret_list

  def show(self) -> None:
    print(f'"{self.name}" has {len(self.network)} layers:')
    for i, layer in enumerate(self.network):
      print(f"  Layer {i+1} has has {len(layer)} nodes called: ", end="")
      for j, node in enumerate(layer):
        if j == 0: print(f"{node.get_name()}", end = "")
        else: print(f", {node.get_name()}", end = "")
      print("")

  def get_data(self) -> str:
    ret_str = ""
    ret_str = str(len(self.network[0]))+"\n"
    for layer in self.network[1:]:
      ret_str+=">\n"
      for neuron in layer:
        ret_str+="-> "+str(neuron.get_data()) + "\n"
    return ret_str[:-1]
  
  def save(self, location: str = "network.txt") -> None:
    with open(NETWORK_STORAGE_LOCATION + location, "w") as f:
      f.write(self.get_data())
  
  def load(self, location: str = "network.txt") -> None:
    with open(NETWORK_STORAGE_LOCATION + location, "r") as f:
      layers = f.read().split(">\n")
      #print(layers)
      self.network = [[] for _ in range(len(layers))]
      self.network[0] = [BaseNeuron(f"I.{i}") for i in range(int(layers[0]))]
      for i, layer in enumerate(layers[1:]):
        neurons = layer.split("\n-> ")
        neurons[0] = neurons[0][3:]
        neurons[-1] = neurons[-1][:-1]
        #input(neurons)
        for j, neuron in enumerate(neurons):
          weights, biases = neuron.split("], [")
          weights = weights[2:].split(", ")
          weights = [float(weight) for weight in weights]
          biases = biases[:-2].split(", ")
          biases = [float(bias) for bias in biases]
          self.network[i+1].append(ProcessNeuron(f"P.{i}.{j}"))
          self.network[i+1][-1].set_data(self.network[i], weights, biases)


def make_data(size: int):
  ret_arr = [[0.0, 0.0, 0.0] for _ in range(size)]
  for i in range(size):
    ret_arr[i][0] = random.random() * 2
    ret_arr[i][1] = random.random() * 2
    ret_arr[i][2] = random.random() * 2
  return ret_arr

def basic_func(input_activations: list[float]) -> list[float]: # Takes in the input activations and returns what the output activations should be
  return True if sum(input_activations) < len(input_activations) else False

def cost(net: Network, learn_func, inputs_tested: list, debug: bool = False):
  cost = 0.0
  for input in inputs_tested:
    true_out = learn_func(input)
    net.set_inputs(input)
    net_out = net.get_outputs()
    cost += abs(net_out[0] - true_out)
  return cost

os.system("cls")
inputs = [[0.1997272259066485, 0.5774567131049464, 1.3128604664088586], [1.2868970183416568, 0.2215832757125702, 0.12200361958165407], [0.6464137959379368, 0.4850734344140377, 1.2540458341158414], [0.3755134471337591, 1.3084826737934423, 1.77768150421768], [1.377082890519213, 1.9915983869530005, 1.618730425511057], [0.1866674974611322, 1.7395521303136887, 0.3430640948119632], [0.29937945873272764, 1.8371342754828819, 1.2915835428419615], [1.7475105472402475, 0.654716080928746, 0.74625314777015], [0.8684580083116444, 0.8058723490027326, 1.384474996754525], [0.2751058233963559, 0.6636874848451941, 1.7601387825266732], [1.7279699890218014, 0.5318847709154637, 0.9317373919825991], [1.8840478452864569, 1.2486985381284474, 1.4299946568175366], [1.8124006235706025, 1.5732607001264833, 1.396747087409707], [1.862537312968793, 1.9846375389104256, 0.5167947191573579], [0.46403645592365916, 1.2755127541166806, 0.4943911068050044], [1.0101716816620596, 1.2971618848625621, 1.9193221070196875], [1.542736280097029, 0.7458292634746322, 1.5383787679224517], [1.7231147588432358, 1.0215636902077958, 1.1313551350694628], [0.45833834023013953, 1.609624972157911, 1.6292766622653612], [1.2690426123010603, 0.5871223752795574, 0.0364015624574463]]
inp = input("Load the last saved network? (y/n) ")
if inp.lower() == "y":
  MyNetwork = Network("MyNetwork", [1, 1])
  MyNetwork.load()
else:
  MyNetwork = Network("MyNetwork", [3, 5, 1])
print(cost(MyNetwork, basic_func, inputs))
inp = input("Save the network? (y/n) ")
if inp.lower() == "y":
  MyNetwork.save()