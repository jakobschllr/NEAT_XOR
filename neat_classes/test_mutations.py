from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker

hist_marker = HistoricalMarker(3)

def test_mutations():
    
    network_1 = Network(2,1)
    network_1.initialize_minimal_network(hist_marker)
    
    network_2 = Network(2,1)
    network_2.initialize_minimal_network(hist_marker)

    print(network_1)
    print(network_2)

    network_1.mutate_with_new_neuron(hist_marker)
    network_2.mutate_with_new_neuron(hist_marker)

    print(network_1)
    print(network_2)
