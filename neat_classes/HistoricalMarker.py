from .neurons.input_neuron import Input_neuron
from .neurons.output_neuron import Output_neuron
from .neurons.hidden_neuron import Hidden_neuron
from .neurons.neuron import Neuron

class HistoricalMarker():
    def __init__(self, hidden_neurons_start_id):
        self.next_hidden_neuron_id = hidden_neurons_start_id
        self.hidden_neuron_ids = {}
        self.next_connection_id = 0
        self.global_connection_ids = {}

    def get_global_connection_id(self, neuron_id_1, neuron_id_2):
        key = str(neuron_id_1) + "" + str(neuron_id_2)
        if key not in self.global_connection_ids:
            self.global_connection_ids[key] = self.next_connection_id
            self.next_connection_id += 1
        return self.global_connection_ids[key]
    
    def get_global_neuron_id(self, neuron_before_id, neuron_after_id):
        # hidden neurons get unique id based on the neuron before and after
        key = str(neuron_before_id) + "" + str(neuron_after_id)
        if key not in self.hidden_neuron_ids:
            self.hidden_neuron_ids[key] = self.next_hidden_neuron_id
            self.next_hidden_neuron_id += 1
        return self.hidden_neuron_ids[key]
