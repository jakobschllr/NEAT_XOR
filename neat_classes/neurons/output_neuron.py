from neat_classes.neurons.neuron import Neuron

class Output_neuron(Neuron):
    def __init__(self, id):
        super().__init__(id, 1)