from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker
from .species import Species
import json
import copy

with open("config.json", "r") as f:
    config = json.load(f)

c1 = config["c1"] # parameter to adjust influence of adjoint gene amount in compatibilty distance calculation
c2 = config["c2"] # parameter to adjust influence of excess gene amount in compatibilty distance calculation
c3 = config["c3"] # parameter to adjust influence of average weight difference in compatibilty distance calculation
species_survival_rate = config["species_survival_rate"] # percentage of fittest networks in each species that survive each generation
elit_nets_amount = config["elit_nets_amount"] # amount of fittest networks, that are elit and are copied to the next generation
mutation_offspring_rate = config["mutation_offspring_rate"] # amount of species children that are created by mutations
crossover_offspring_rate = config["crossover_offspring_rate"] # amount of species children that are created by crossovers
weight_perturbation_prob = config["weight_perturbation_prob"] # chance of a connection weight being mutated with perturbation
weight_mutation_random_value_prob = config["weight_mutation_random_value_prob"] # chance of a connection weight being mutated by setting it to a random value
bias_weight_mutation_prob = config["bias_weight_mutation_prob"] # chance of a bias in a random neuron in a network being perturbarted

gene_disabled_rate = config["gene_disabled_rate"] # chance of inherited connection staying disabled if it is disabled in either parent (crossover)
new_neuron_prob = config["new_neuron_prob"] # chance a new hidden neuron is added to a network
new_connection_prob = config["new_connection_prob"] # chance of a new connectino being added to a network
species_similarity_threshold = config["species_similarity_threshold"] # threshold for compatibilty between networks (the higher the treshold the more networks are in a species)
stagnation_treshold = config["stagnation_treshold"] # amount of generations a species survives without fitness improvements

class PopulationHandler():
    def __init__(self, initial_network, input_neurons, output_neurons):
        self.network_amount = initial_network
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
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

        initial_species.calculate_fitness()
        # calculate adjusted fitness for each network in first species and remove low performing
        self.species.append(initial_species)
        print(f"Initial Species with {self.network_amount} organisms populated")
                

    def start_evolution_process(self):
        generation_counter = 0
        best_network = None
        while best_network == None or best_network.raw_fitness < 0.9:
            # get sum of all average adjusted fitnesses of all species
            print("********** Generation ", generation_counter, " **********")
            adjusted_fitness_all_species = 0
            for i in range(len(self.species)):

                #adjust stagnation counter for species
                self.species[i].adjust_stagnation_counter()

                #if species fitness stagnated to long remove species
                if self.species[i].stagnation_counter == stagnation_treshold:
                    self.species[i] = None
                    print("Removed species due to stagnation")

                else:
                #remove low performing networks
                    self.species[i].remove_lowperforming_networks()
                    adjusted_fitness_all_species += self.species[i].total_adjusted_fitness       

            self.species = [species for species in self.species if species is not None]

            new_population = []

            # each species produces certain amount of children based on its adjusted fitness

            for species in self.species:
                
                species_children_amount = round((species.total_adjusted_fitness / adjusted_fitness_all_species) * self.network_amount)
                children, elit_networks = species.produce_offspring(
                    species_children_amount,
                    mutation_offspring_rate,
                    weight_perturbation_prob,
                    weight_mutation_random_value_prob,
                    bias_weight_mutation_prob,
                    crossover_offspring_rate,
                    new_neuron_prob,
                    new_connection_prob,
                    self.hist_marker,
                    elit_nets_amount,
                    gene_disabled_rate
                )

                if children != None and elit_networks != None:
                    #print("Species created ", len(children), " children")
                    new_population = new_population + children
                    for net in elit_networks:
                        if best_network == None or net.raw_fitness > best_network.raw_fitness:
                            best_network = copy.deepcopy(net)

            # reset species
            for species in self.species:
                species.reset_species()

            # all newly produced children have to be assigned to a species based on their compatibilty distance
            for network in new_population:
                found_species = False

                for species in self.species:
                    compatibility_dist = species.calculate_compatibility_distance(network)
                    #print("Network ", network, " has compatibility distance ", compatibility_dist)
                    if compatibility_dist < species_similarity_threshold:
                        species.add_network(network)
                        found_species = True
                        break
                if not found_species:
                    # create new species
                    new_species = Species(c1, c2, c3, species_survival_rate)
                    #print("Network ", network, " added to new Species because ", compatibility_dist,  " > ", species_similarity_threshold)
                    new_species.add_network(network)
                    self.species.append(new_species)

            # nicht verwendete Spezien löschen
            self.species = [s for s in self.species if len(s.networks) > 0]
            
            # update average adjusted fitness in each species
            for species in self.species:
                species.calculate_fitness()
                species.update_representative()
                
                print("----Species----")
                print(species)

            print("Population size: ", len(new_population))

            generation_counter += 1

        best_network.reset_neurons()
        print("Best network: ", best_network)
        print("Input   |   Raw Output   |   Rounded Ouput")
        print(f" 0 0    |       {round(best_network.compute_inputs(0,0), 2)}     |   {round(best_network.compute_inputs(0,0))}")
        print(f" 0 1    |       {round(best_network.compute_inputs(0,1), 2)}     |   {round(best_network.compute_inputs(0,1))}")
        print(f" 1 0    |       {round(best_network.compute_inputs(1,0), 2)}     |   {round(best_network.compute_inputs(1,0))}")
        print(f" 1 1    |       {round(best_network.compute_inputs(1,1), 2)}     |   {round(best_network.compute_inputs(1,1))}")
