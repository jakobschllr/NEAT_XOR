from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker
from .species import Species
from .neurons.neuron import Neuron

hist_marker = HistoricalMarker(3)
c1 = 1.0
c2 = 1.0
c3 = 0.4
species_survival_rate = 20

def test_mutations():
    
    # initial_species = Species(c1, c2, c3, species_survival_rate)
    # for _ in range(0, 150):
    #     network = Network(2, 1)
    #     network.initialize_minimal_network(hist_marker)
    #     initial_species.add_network(network)


    # initial_species.calculate_total_adjusted_fitness()
    # initial_species.remove_lowperforming_networks()
    
    # for net in initial_species.networks:
    #     print("Network raw fitness: ", net.raw_fitness, " Network adjusted fitness: ", net.adjusted_fitness)

    neuron = Neuron(0)

    print("Sigmoid(0) = ", neuron.calculate_output(0, "modified-sigmoid"))  # Sollte ≈ 0.5 sein
    print("Sigmoid(5) = ", neuron.calculate_output(5, "modified-sigmoid"))  # Sollte ≈ 1.0 sein
    print("Sigmoid(-5) = ", neuron.calculate_output(-5, "modified-sigmoid"))  # Sollte ≈ 0.0 sein
