from .neurons.input_neuron import Input_neuron
from .neurons.output_neuron import Output_neuron
from neat_classes.HistoricalMarker import HistoricalMarker
from .neurons.hidden_neuron import Hidden_neuron
from .neurons.neuron import Neuron
from .connection import Connection
from queue import Queue
import random


class Network():
    def __init__(self, input_neurons_amount, output_neurons_amount):
        self.input_neurons_amount = input_neurons_amount
        self.output_neurons_amount = output_neurons_amount
        self.input_neurons = []
        self.output_neurons = []
        self.hidden_neurons = []
        self.connections: list[Connection] = []
        self.perturbation_strength = 0.01
    
        self.raw_fitness = None
        self.adjusted_fitness = None

    def initialize_minimal_network(self, hist_marker: HistoricalMarker):
        initial_neuron_id_counter = 0
        for _ in range(0, self.input_neurons_amount):
            in_neuron = Input_neuron(initial_neuron_id_counter)
            self.input_neurons.append(in_neuron)
            initial_neuron_id_counter += 1

        for _ in range(0, self.output_neurons_amount):
            out_neuron = Output_neuron(initial_neuron_id_counter)
            self.output_neurons.append(out_neuron)      
            initial_neuron_id_counter += 1 

        for in_neuron in self.input_neurons:
            for out_neuron in self.output_neurons:
                self.add_new_connection(in_neuron, out_neuron, hist_marker)

    def mutate_with_new_neuron(self, hist_marker: HistoricalMarker):
        # zufällige aktive Connection auswählen, die unterbrochen werden soll und diese disablen
        connection = random.choice([conn for conn in self.connections if conn.enabled])
        connection.disable()
        neuron_before = connection.neuron_in
        neuron_after = connection.neuron_out

        # create new hidden neuron
        hidden_neuron = Hidden_neuron(hist_marker, neuron_before, neuron_after)

        # create two new connections 
        self.add_new_connection(neuron_before, hidden_neuron, hist_marker)
        self.add_new_connection(hidden_neuron, neuron_after, hist_marker)

        # neues Hidden Neuron und neue Connections speichern
        self.hidden_neurons.append(hidden_neuron)

    def mutate_with_new_connection(self, hist_marker: HistoricalMarker): # create new random connection in network
        # add new connection between existing neurons (but without connection input neurons, without creating cycles)
        all_possible_connections = []

        # collect all possible connections
        for in_neuron in self.input_neurons:
            for out_neuron in self.output_neurons:
                all_possible_connections.append([in_neuron.id, out_neuron.id])
        for hidden_neuron in self.hidden_neurons:
            for out_neuron in self.output_neurons:
                all_possible_connections.append([hidden_neuron.id, out_neuron.id])

        for in_neuron in self.input_neurons:
            for hidden_neuron in self.hidden_neurons:
                all_possible_connections.append([in_neuron.id, hidden_neuron.id])

        for hidden_neuron_1 in self.hidden_neurons:
            for hidden_neuron_2 in self.hidden_neurons:
                all_possible_connections.append([hidden_neuron_1.id, hidden_neuron_2.id])


        # remove already existing connections (only enabled ones)
        for connection in [c for c in self.connections if c.enabled]:
            existing_conn = [connection.neuron_in.id, connection.neuron_out.id]
            if existing_conn in all_possible_connections:
                all_possible_connections.remove(existing_conn)
        

        # remove swapped connections to avoid cycles
        for connection in [c for c in self.connections if c]:
            swapped_conn = [connection.neuron_out.id, connection.neuron_in.id]
            if swapped_conn in all_possible_connections:
                all_possible_connections.remove(swapped_conn)

        # remove connections of neurons to themselves
        for connection in all_possible_connections:
            if connection[0] == connection[1]:
                all_possible_connections.remove(connection)


        if len(all_possible_connections) == 0:
            return
        else:
            # choose random connection from those who aren't active or non-existent
            random_connection = random.choice(all_possible_connections)
            
            # if new connection already exists but is disabled, enable it again
            for conn in self.connections:
                if not conn.enabled and (random_connection == [conn.neuron_in.id, conn.neuron_out.id]):
                    conn.enable()
                    return

            # connection does not exist, create it
            neuron_in = [n for n in self.get_all_neurons() if n.id == random_connection[0]][0]
            neuron_out = [n for n in self.get_all_neurons() if n.id == random_connection[1]][0]
            self.add_new_connection(neuron_in, neuron_out, hist_marker)


    # add new connection between two neurons
    def add_new_connection(self, neuron_1: Neuron, neuron_2: Neuron, hist_marker: HistoricalMarker):
        self.connections.append(Connection(neuron_1, neuron_2, hist_marker))

    def add_existing_connection(self, connection: Connection):
        self.connections.append(connection)

    def get_input_neurons(self):
        return self.input_neurons
    
    def get_output_neurons(self):
        return self.output_neurons
    
    def reset_neurons(self):
        for neuron in self.get_all_neurons():
            neuron.reset()
    
    def compute_inputs(self, *inputs):
        queue = Queue()
        input_idx = 0
        input_neurons = self.get_input_neurons()
        for input_neuron in input_neurons:
            input_neuron.add_weighted_input(inputs[input_idx])
            queue.put(input_neuron)
            input_idx += 1

        visited_neurons = []

        while queue.empty() == False:
            current_neuron = queue.get()
            visited_neurons.append(current_neuron)
            
            # Summe der gewichteten Eingaben berechnen
            weighted_sum = current_neuron.get_weighted_bias()
            for weighted_input in current_neuron.current_weighted_inputs:
                weighted_sum += weighted_input

            # rohe Ausgabe mit Aktivierungsfunktion berechnen
            raw_output = current_neuron.calculate_output(weighted_sum, "modified-sigmoid")
            current_neuron.raw_output = raw_output

            # an jeden Nachfolger den gewichteten Wert weitergeben
            connections = [conn for conn in self.connections if conn.starts_with_neuron(current_neuron) and conn.enabled]

            if len(connections) > 0:
                for connection in connections:
                    weighted_value = connection.weight * raw_output
                    child = connection.neuron_out
                    child.add_weighted_input(weighted_value)
                    if child not in visited_neurons:
                        queue.put(child)

        final_output = self.output_neurons[0].raw_output
        self.reset_neurons()

        return final_output
    
    def apply_random_weight_mutation(self):
        connection = random.choice(self.connections)
        connection.set_random_weight()

    def apply_bias_weight_mutation(self):
        neurons = self.get_all_neurons()
        neuron = random.choice(neurons)
        neuron.mutate_bias_weight()

    def apply_weight_perturbation(self):
        random_value = random.uniform(0.0, 1.0)
        if random_value < 0.5:
            perturbation_value = self.perturbation_strength
        else:
            perturbation_value = - self.perturbation_strength

        connection = random.choice(self.connections)
        connection.weight = connection.weight + perturbation_value


    def __str__(self):
        output = "Network: "
        # output += "Connections: "
        # for connection in self.connections:
        #     output += str(connection)
        # output += "Input-Neurons: "
        # for neuron in self.input_neurons:
        #     output += str(neuron.id) + " "
        # output += "Hidden-Neurons: "
        # for neuron in self.hidden_neurons:
        #     output += str(neuron.id) + " "
        # output += "Output-Neurons: "
        # for neuron in self.output_neurons:
        #     output += str(neuron.id) + " "
        
        output += "Raw Fitness: " + str(self.raw_fitness)

        return output
    
    def get_all_neurons(self):
        return self.input_neurons + self.hidden_neurons + self.output_neurons