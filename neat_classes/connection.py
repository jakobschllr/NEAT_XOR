from neat_classes.neurons.neuron import Neuron
from neat_classes.HistoricalMarker import HistoricalMarker
import random


class Connection():
    def __init__(self, neuron_in: Neuron, neuron_out: Neuron, hist_marker: HistoricalMarker):
        self.neuron_in = neuron_in
        self.neuron_out = neuron_out
        self.weight = random.uniform(-20.0, 20.0)
        self.enabled = True
        self.connection_id = hist_marker.get_global_connection_id(neuron_in.id, neuron_out.id)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_random_weight(self):
        self.weight = random.uniform(-20.0, 20.0)

    def starts_with_neuron(self, neuron: Neuron):
        return self.neuron_in.id == neuron.id
    
    def __str__(self):
        return ("|" + str(self.neuron_in.id) + "->" + str(self.neuron_out.id) + "| ")