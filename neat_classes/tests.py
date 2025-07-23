from .network import Network
from .connection import Connection
from neat_classes.HistoricalMarker import HistoricalMarker
from .species import Species
from .neurons.neuron import Neuron
from .neurons.hidden_neuron import Hidden_neuron
import math

hist_marker = HistoricalMarker(3)
c1 = 1.0
c2 = 1.0
c3 = 0.4
species_survival_rate = 20
elit_nets_amount = 2 

mutation_offspring_rate = 50 
weight_mutation_perturbation_rate = 20
weight_mutation_random_value_rate = 5
bias_weight_mutation_rate = 10

crossover_offspring_rate = 50 

new_neuron_probability = 10 
new_connection_probability = 15

gene_disabled_rate = 75

def test_mutations():
    pass

def test_network_forward_pass():

    net = Network(2,1)

    net.initialize_minimal_network(hist_marker)
    net.connections = []
    
    hidden1 = Hidden_neuron(hist_marker, net.input_neurons[0], net.output_neurons[0])
    hidden1.bias_weight = -10
    hidden2 = Hidden_neuron(hist_marker, net.input_neurons[1], net.output_neurons[0])
    hidden2.bias_weight = 20

    net.hidden_neurons.append(hidden1)
    net.hidden_neurons.append(hidden2)

    i1_to_hidden1 = Connection(net.input_neurons[0], net.hidden_neurons[0], hist_marker)
    i2_to_hidden1 = Connection(net.input_neurons[1], net.hidden_neurons[0], hist_marker)
    i1_to_hidden1.weight = 15
    i2_to_hidden1.weight = 15

    i1_to_hidden2 = Connection(net.input_neurons[0], net.hidden_neurons[1], hist_marker)
    i2_to_hidden2 = Connection(net.input_neurons[1], net.hidden_neurons[1], hist_marker)
    i1_to_hidden2.weight = -15
    i2_to_hidden2.weight = -15

    hidden1_to_out = Connection(net.hidden_neurons[0], net.output_neurons[0], hist_marker)
    hidden2_to_out = Connection(net.hidden_neurons[1], net.output_neurons[0], hist_marker)
    hidden1_to_out.weight = 10
    hidden2_to_out.weight = 10

    net.output_neurons[0].bias_weight = -15

    net.connections.append(i1_to_hidden1)
    net.connections.append(i2_to_hidden1)
    net.connections.append(i1_to_hidden2)
    net.connections.append(i2_to_hidden2)
    net.connections.append(hidden1_to_out)
    net.connections.append(hidden2_to_out)

    print(net)

    net.compute_inputs(0,0)
    inputs = [(0,0), (0,1), (1,0), (1,1)]
    for x1, x2 in inputs:
        y = net.compute_inputs(x1, x2)
        print(f"XOR({x1}, {x2}) = {y}")