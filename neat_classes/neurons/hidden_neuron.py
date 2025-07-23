from neat_classes.neurons.neuron import Neuron

class Hidden_neuron(Neuron):
    def __init__(self, historcial_marker, neuron_before, neuron_after):
        self.neuron_before = neuron_before
        self.neuron_after = neuron_after
        super().__init__(historcial_marker.get_global_neuron_id(neuron_before.id, neuron_after.id), 1)