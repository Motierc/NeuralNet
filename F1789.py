import random
import os
import math
import copy

NETWORK_STORAGE_LOCATION = r"[INSERT LOCATION HERE]"


def same(x: float) -> float:
  return x

def d_same(_) -> int:
  return 1

def relu(x: float | int) -> float | int:
  return max(0, x)

def d_relu(x) -> float: 
  return 0 if x<0 else 1

def sigmoid(x):
  sig = 1 / (1 + math.exp(-x))
  return sig

def d_sigmoid(x):
  return math.exp(-x)/(1+math.exp(-x))^2

def cented_outs(inps: list[float]):
  total = sum(inps)
  return [(inp / total * 100) for inp in inps]

ACTIVATION_FUNCTIONS = {
  "same": d_same,
  "relu": d_relu,
  "sigmoid": d_sigmoid
}

def get_activation_func(func_name):
  if func_name == "same": return same
  elif func_name == "relu": return relu
  elif func_name == "sigmoid": return sigmoid
  else: raise ValueError(f"[{func_name}] doesn't exist!")


def make_data(size: int) -> list[list[float]]:
  return [[(random.random()), (random.random())] for _ in range(size)]

def baby_func(input_activations: list[float]) -> list[float]:
  if input_activations[0] > 0: return [1]
  else: return[-1]

def basic_func(input_activations: list[float]) -> list[float]: # Takes in the input activations and returns what the output activations should be
  return [1, 0] if input_activations[0] < input_activations[1] else [0, 1]

def neuron_cost(out_act: float, exp_out: float) -> float:
  return (out_act-exp_out)**2

def d_neuron_cost(out_act: float, exp_out: float) -> float:
  return 2 * (out_act - exp_out)


class BaseNeuron: # Used for process neuron and for input neuron
  def __init__(self, name: str = "", activation_function = same, activation: float = 0) -> None:
    self.name = name
    self.activation = activation
    self.activation_function = activation_function

  def set_activation(self, activation: float) -> None:
    self.activation = self.activation_function(activation)

  def get_activation(self) -> float: 
    return self.activation
  
  def show(self) -> tuple[str, float]:
    return self.name, self.activation
  
  def get_name(self) -> str: 
    return self.name

  def get_d_act_func(self):
    return ACTIVATION_FUNCTIONS[self.activation_function.__name__]

class ProcessNeuron(BaseNeuron): # main neurons in the neural network
  def __init__(self, name: str, activation_function = same) -> None:
    self.attached_neurons: list[BaseNeuron]
    self.attached_neurons = []
    self.weights: list[float]
    self.biases: list[float]
    self.activation = 0
    self.activation: float
    self.name = name
    self.activation_function = activation_function

  def attach(self, neurons: list[BaseNeuron]) -> None:
    self.attached_neurons = neurons
    self.weights = [(random.random()-0.5) * 2 for _ in range(len(neurons))]
    self.biases = [(random.random()-0.5) * 2 for _ in range(len(neurons))]

  def add_neuron(self, neurons: list) -> None:
    for neuron in neurons:
        self.attached_neurons.append(neuron)
        self.weights = [(random.random()-0.5) * 2 for _ in range(len(neurons))]
        self.biases = [(random.random()-0.5) * 2 for _ in range(len(neurons))]

  def calculate_activation(self) -> None:
    self.activation = 0
    neuron_activations = [neuron.get_activation() for neuron in self.attached_neurons]
    for neuron_activation, neuron_weight, neuron_bias in zip(neuron_activations, self.weights, self.biases):
      self.activation += (neuron_activation * neuron_weight) + neuron_bias
    self.activation = self.activation_function(self.activation)

  def get_activation(self) -> float:
    self.calculate_activation()
    return self.activation
  
  def show(self, display: bool = True) -> list[str, list[str], list[float], list[float]]:
    nod_names = [neuron.get_name() for neuron in self.attached_neurons]
    if display:
      print(f'"{self.name}" (with activation {self.activation}) is connected to {len(nod_names)} neuron(s):')
      for i in range(len(nod_names)):
        print(f'  {i+1}) "{nod_names[i]}" with a weight of {self.weights[i]} and bias of {self.biases[i]}')
    return [self.name, nod_names, self.weights, self.biases]

  def get_data(self) -> list[float]:
    return [self.weights, self.biases, self.activation_function.__name__]

  def set_data(self, neurons: list[BaseNeuron], weights: list[float], biases: list[float], activation_function):
    self.attached_neurons = neurons
    self.weights = weights
    self.biases = biases
    self.activation_function = activation_function

class Network:
  def __init__(self, name: str, network_structure: list[int], activation_functions: list = None) -> None:
    self.name = name
    self.network: list[list[BaseNeuron]]
    self.network = [[] for _ in range(len(network_structure))]
    if activation_functions is not None and activation_functions[0] is not None:
      print(f"Inputs using: {activation_functions[0]}")
      self.network[0] = [BaseNeuron(f"I.{i}", activation_functions[0]) for i in range(network_structure[0])]
    else:
      self.network[0] = [BaseNeuron(f"I.{i}") for i in range(network_structure[0])]
    for i, layer_size in enumerate(network_structure[1:]): # All the layers except for input layer
      if activation_functions is not None and activation_functions[i+1] is not None:
        print(f"P.{i} using: {activation_functions[i+1]}")
        self.network[i+1] = [ProcessNeuron(f"P.{i}.{j}", activation_functions[i+1]) for j in range(layer_size)]
      else:
        self.network[i+1] = [ProcessNeuron(f"P.{i}.{j}") for j in range(layer_size)]
      for neuron in self.network[i+1]:
        neuron.attach(self.network[i])

  def set_inputs(self, inputs: list[float]) -> None:
    for neuron, input in zip(self.network[0], inputs):
      neuron.set_activation(input)

  def get_outputs(self, set_to: list[float] = None) -> list[float]:
    if set_to is not None: self.set_inputs(set_to)
    ret_list = [out_neuron.get_activation() for out_neuron in self.network[-1]]
    return ret_list

  def show(self) -> None:
    print(f'"{self.name}" has {len(self.network)} layers:')
    for i, layer in enumerate(self.network):
      print(f"  Layer {i+1} has has {len(layer)} neurons called: ", end="")
      for j, neuron in enumerate(layer):
        if j == 0: print(f"{neuron.get_name()}", end = "")
        else: print(f", {neuron.get_name()}", end = "")
      print("")

  def get_data(self) -> str:
    ret_str = f"{str(len(self.network[0]))}, {self.network[0][0].activation_function.__name__}'\n"
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
      inp_neuron_cnt, inp_act_func = layers[0].split(", ")
      self.network[0] = [BaseNeuron(f"I.{i}", get_activation_func(inp_act_func[:-2])) for i in range(int(inp_neuron_cnt))]
      for i, layer in enumerate(layers[1:]):
        neurons = layer.split("\n-> ")
        neurons[0] = neurons[0][3:]
        neurons[-1] = neurons[-1][:-2]
        for j, neuron in enumerate(neurons):
          weights, biases = neuron.split("], [")
          weights = weights[2:].split(", ")
          weights = [float(weight) for weight in weights]
          biases = biases.split(", ")
          activation_function = biases[-1].replace("[", "").replace("]", "").replace("'", "")
          biases = biases[:-1]
          #activation_function = same
          activation_function = get_activation_func(activation_function)
          biases[-1] = biases[-1][:-1]
          biases = [float(bias) for bias in biases]
          self.network[i+1].append(ProcessNeuron(f"P.{i}.{j}"))
          self.network[i+1][-1].set_data(self.network[i], weights, biases, activation_function)

  def calc_out_layer_neuron_vals(self, exp_outs) -> list[float]: # black magic that does magic
    neuron_vals = [0.0 for _ in range(len(exp_outs))]
    for i in range(len(exp_outs)):
      d_cost = d_neuron_cost(self.network[-1][i].activation, exp_outs[i])
      d_act_func = self.network[-1][i].get_d_act_func()(self.network[-1][i].activation)
      neuron_vals[i] = d_cost * d_act_func
    return neuron_vals

def cost(net: Network, func_to_learn, inputs_tested: list[list], debug: bool = False):
  cost = 0.0
  #print(len(inputs_tested))
  for input in inputs_tested:
    true_outs = func_to_learn(input)
    net.set_inputs(input)
    net_outs = net.get_outputs()
    for net_out, true_out in zip(net_outs, true_outs):
      cost += neuron_cost(net_out, true_out)
  return cost / len(inputs_tested)

def back_propagation(net: Network, func_to_learn, inputs_to_test: list[list], batch_size: int = 20, batches: int = 1, epochs: int = 1, learn_rate: float = 0.05):
  start_cost = cost(net, func_to_learn, inputs_to_test)
  #Makes a cost grad that goes layer[neuron[weight]]: list[list[float]]
  for z in range(epochs):
    print(f"{z+1}/{epochs} epochs started. Current cost = {cost(net, func_to_learn, inputs_to_test)}")
    for _ in range(batches):
      cost_grad_w = [[] for _ in range(len(net.network)-1)]
      for i, layer in enumerate(net.network[1:]):
        cost_grad_w[i] = [[] for _ in range(len(layer))]
        for j in range(len(layer)):
          cost_grad_w[i][j] = [0.0 for _ in range(len(net.network[i]))]
      cost_grad_w: list[list[float]]

      cost_grad_b = copy.deepcopy(cost_grad_w) # Has the same layout as the weigths
      #print(f"cost {i} = {cost(net, func_to_learn, inputs_to_test)}")
      inputs = make_data(batch_size)
      for inp in inputs:
        net.get_outputs(inp)
        exp_out = func_to_learn(inp)

        neuron_vals = [[] for _ in range(len(net.network)-1)]
        for i, layer in enumerate(net.network[1:]):
          neuron_vals[i] = [0.0 for _ in range(len(layer))]
        neuron_vals: list[list[float]]
        neuron_vals[-1] = net.calc_out_layer_neuron_vals(exp_out)

        for i, neuron in enumerate(net.network[-1]): # last layer neurons
          neuron: ProcessNeuron
          for j, att_neuron in enumerate(neuron.attached_neurons):
            cost_grad_w[-1][i][j] += att_neuron.activation * neuron_vals[-1][i]
            cost_grad_b[-1][i][j] += 1 * neuron_vals[-1][i]
        
        for i, layer in reversed(list(enumerate(net.network[1:-1]))):
          prev_layers_weights = [neuron.weights for neuron in net.network[i+2]]
          for j, neuron in enumerate(layer):
            for k, weight in enumerate(prev_layers_weights):
              neuron_vals[i][j] += weight[j] * neuron_vals[i + 1][k]
            neuron_vals[i][j] *= neuron.get_d_act_func()(neuron.activation)
            for k, att_neuron in enumerate(neuron.attached_neurons):
              cost_grad_w[i][j][k] += neuron_vals[i][j] * att_neuron.activation
              cost_grad_b[i][j][k] += neuron_vals[i][j]

      for i, layer in enumerate(net.network[1:]):
        for j, neuron in enumerate(layer):
          for k in range(len(neuron.weights)):
            neuron.weights[k] -= (cost_grad_w[i][j][k] / batch_size) * learn_rate 
          for k in range(len(neuron.biases)):
            neuron.biases[k] -= (cost_grad_b[i][j][k] / batch_size) * learn_rate 
  end_cost = cost(net, func_to_learn, inputs_to_test)
  print(f"\nEnd cost = {end_cost} v.s start cost = {start_cost} which meanse the network got {round((start_cost-end_cost)/end_cost*100, 3)}% better\n")

def prettify_outs(num1, num2, net: Network):
  net_out = cented_outs(net.get_outputs([num1, num2]))
  print(f"The network believes {num1} > {num2} being True is {round(net_out[1],3)}% likely while it being False is {round(net_out[0],3)}%")

os.system("cls")

# <-- EXAMPLE USE --> #

inp = input("Load the last saved network? (y/n) ")
if inp.lower() == "y":
  MyNetwork = Network("MyNetwork", [1, 1])
  MyNetwork.load()
else:
  MyNetwork = Network("MyNetwork", [2, 2, 3, 2], [sigmoid, None, None, relu])

#print(make_data(100))

inputs = [[0.5545381008701183, 0.7601236784427154], [0.38641988584327225, 0.8198671211521595], [0.40328719880160835, 0.27681049912035194], [0.8027061461107167, 0.5550826937433313], [0.22636992580494508, 0.4787241990497656], [0.9397314087612971, 0.6674555152530243], [0.7852354656181761, 0.2239334653988062], [0.09218263574995789, 0.11173562670170167], [0.32554234686325667, 0.6546122499668604], [0.9335103954376316, 0.358280139619903], [0.8802480340514433, 0.5937981050480791], [0.8345064140217353, 0.6836586319958008], [0.31184306496867453, 0.8609615202690697], [0.238294580252144, 0.4655756609546706], [0.8438633447387087, 0.2949035246697296], [0.6652980667014368, 0.15125918055698329], [0.8017195784838427, 0.035101368907551556], [0.6400453062507313, 0.7498161519500126], [0.6237468936060517, 0.6885099922379551], [0.8111445146607877, 0.23273027578908745], [0.7788548184973016, 0.24527348851778463], [0.028382955092771778, 0.8460917068841928], [0.7931409565102431, 0.20230235945572383], [0.3213850527299593, 0.6028547179725939], [0.2471348191622067, 0.4273358323325961], [0.41850083725514775, 0.2457591836052585], [0.7377899062882011, 0.4378839429534923], [0.9441500527407678, 0.5898047906179058], [0.17974142629976364, 0.19751405630843077], [0.331369052161231, 0.36853072740271964], [0.796770421445333, 0.8345313868485983], [0.33340814825687515, 0.8522828201413545], [0.3744763753221967, 0.3766883447672186], [0.16317510990016204, 0.5377212968983115], [0.8769531264095357, 0.1830731477459776], [0.7202184558886056, 0.5575902053240621], [0.88174229250349, 0.17807149582668158], [0.18745096251109017, 0.8736358991395743], [0.05773788239477762, 0.20865052405286322], [0.9472539280998461, 0.6967237798361234], [0.7108388562430319, 0.959060984183587], [0.4643132628192068, 0.6783165508216807], [0.8239297637894946, 0.4813610365124381], [0.30566503515260135, 0.7727585312519363], [0.6257894063895632, 0.3411238963266695], [0.03164387676181102, 0.707931655462226], [0.11276616131826489, 0.07341153016403767], [0.8777193802816673, 0.4598332613198399], [0.05926628467117401, 0.7187120893660157], [0.1911478822919216, 0.8797575504552495], [0.9292520460255078, 0.8236923710209592], [0.5665295113819243, 0.7956717060298749], [0.3423702493424092, 0.9589538547150794], [0.03419706977491466, 0.2935548788258606], [0.22170379245602967, 0.01866314912261058], [0.46316735196362, 0.6362711980697966], [0.4889163104356342, 0.7621100216829999], [0.26645007971112444, 0.6668292137478695], [0.08603713431712756, 0.864224905016153], [0.19709377900070935, 0.33331166376265287], [0.16373677879623527, 0.2019423848764903], [0.24817567011948882, 0.17510489846285127], [0.12965042018663864, 0.738580094966507], [0.1456113843135769, 0.16464897536233225], [0.4754381559254005, 0.06106291389716012], [0.009038933107247127, 0.647041383854352], [0.11392570314601413, 0.9345799722558671], [0.28281557132827395, 0.9064199488379799], [0.802705413285295, 0.7209922910196579], [0.45292429922215827, 0.6616464286050259], [0.851664048953914, 0.9108574823551513], [0.6889227860192231, 0.9940171765484463], [0.6335657462815776, 0.682246213728532], [0.18095060749647562, 0.9970797225110665], [0.7026321793281106, 0.10995735043379562], [0.9198563140546371, 0.4953758099360015], [0.15276299333797017, 0.2629560633618524], [0.9032242284871494, 0.8886429600368188], [0.3243728251644912, 0.8531356997217033], [0.7029128884664549, 0.4243220015220006], [0.7124503086092991, 0.48174442729126943], [0.6167137378732231, 0.34055436517641813], [0.7408496420092343, 0.9461022811529017], [0.47011432356344407, 0.6472132722889511], [0.7311226947007227, 0.6299969382322684], [0.520117406938315, 0.6731507878539148], [0.3931253476937884, 0.15504255247519927], [0.7922330482347517, 0.4878174155134323], [0.19807799669465898, 0.09410541942838002], [0.31466316833496766, 0.37561905544134033], [0.5222970917947651, 0.8249976275137457], [0.060271529006068025, 0.06649980013842527], [0.17315769384504653, 0.8983993256211384], [0.4132224329026063, 0.6031223927629608], [0.8993910745778106, 0.9106661748079152], [0.7283185743181031, 0.6851018058246965], [0.68075060387394, 0.12375240308952196], [0.5237924938148931, 0.4620865583400523], [0.15972202870594965, 0.7204156032552542], [0.8206067202833516, 0.25559906013260925]]
back_propagation(MyNetwork, basic_func, inputs, 100, 100, 20)

prettify_outs(0.3, 0.5, MyNetwork)
prettify_outs(1, 0.3, MyNetwork)
prettify_outs(-100, 100, MyNetwork)
prettify_outs(0.5, 0.5, MyNetwork)
prettify_outs(0.1, 0.2, MyNetwork)

inp = input("\nSave the network? (y/n) ")
if inp.lower() == "y":
  MyNetwork.save()