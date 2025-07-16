from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker
from .species import Species
import random
import time

c1 = 1.0
c2 = 1.0
c3 = 0.4
species_survival_rate = 20

mutation_offspring_rate = 80 # mutation of connection weight
weight_mutation_perturbation_rate = 90
weight_mutation_random_value_rate = 8
bias_weight_mutation_rate = 2

crossover_offspring_rate = 20 
gene_disabled_rate = 75 # gene stays disabled if it was disabled in either parent

new_neuron_probability = 2 # percent
new_connection_probability = 8 # percent

species_similarity_threshold = 0.30 # bestimmt wie ähnlich sich netzwerke sein müssen, um zu selben Spezies zu gehören


class PopulationHandler():
    def __init__(self):
        self.network_amount = 150
        self.input_neurons = 2
        self.output_neurons = 1
        self.values = [0, 1]
        self.species: list[Species] = []
        self.hist_marker = HistoricalMarker(self.input_neurons + self.output_neurons) # hidden neuron IDs start after IDs for input and output neurons

    def initial_population(self):
        # create first species
        initial_species = Species(c1, c2, c3, species_survival_rate)

        # create networks and add to first species
        for _ in range(0, self.network_amount):
            network = Network(self.input_neurons, self.output_neurons)
            network.initialize_minimal_network(self.hist_marker)
            initial_species.add_network(network)

        initial_species.calculate_total_adjusted_fitness()
        # calculate adjusted fitness for each network in first species and remove low performing
        self.species.append(initial_species)
        print(f"Initial Species with {self.network_amount} organisms populated")
                

    def start_evolution_process(self):
        generation_counter = 0
        while generation_counter < 100:
            # get sum of all average adjusted fitnesses of all species
            print("****** Generation ", generation_counter, " ******")
            adjusted_fitness_all_species = 0
            for species in self.species:
                species.remove_lowperforming_networks()
                adjusted_fitness_all_species += species.total_adjusted_fitness        

            new_population = []

            # each species produces certain amount of children based on it's average adjusted fitness
            for species in self.species:
                species_children_amount = round((species.total_adjusted_fitness / adjusted_fitness_all_species) * self.network_amount)
                children = species.produce_offspring(
                    species_children_amount,
                    mutation_offspring_rate,
                    weight_mutation_perturbation_rate,
                    weight_mutation_random_value_rate,
                    bias_weight_mutation_rate,
                    crossover_offspring_rate,
                    new_neuron_probability,
                    new_connection_probability,
                    self.hist_marker
                )

                #print("--> Species produced " + str(len(children)) + " children")
                new_population = new_population + children

                # reset species
                species.reset_species()

            # all newly produced children have to be assigned to a species based on their compatibilty distance
            for network in new_population:
                found_species = False
                for species in self.species:
                    compatibility_dist = species.calculate_compatibility_distance(network)
                    if compatibility_dist <= species_similarity_threshold:
                        species.add_network(network)
                        found_species = True
                        break
                if not found_species:
                    # create new species
                    new_species = Species(c1, c2, c3, species_survival_rate)
                    new_species.add_network(network)
                    self.species.append(new_species)

            # nicht verwendete Spezien löschen
            self.species = [s for s in self.species if len(s.networks) > 0]
            
            # update average adjusted fitness in each species
            for species in self.species:
                species.calculate_total_adjusted_fitness()
            
            print("Species: ")
            for species in self.species:
                print(species)

            generation_counter += 1
            #time.sleep(0.1)