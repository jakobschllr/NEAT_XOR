from .neurons.input_neuron import Input_neuron
from .neurons.output_neuron import Output_neuron
from .neurons.hidden_neuron import Hidden_neuron
from .network import Network
from neat_classes.HistoricalMarker import HistoricalMarker
import math
import random
import copy

class Species():
    def __init__(self, c1, c2, c3, survival_rate):
        self.networks = []
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.survival_rate = survival_rate
        self.representative = None
        self.children_amount_next_generation = 0
        self.total_adjusted_fitness = 0

    def add_network(self, network: Network):
        self.networks.append(network)
        if self.representative == None:
            self.representative = network
        elif network.raw_fitness != None and network.raw_fitness > self.representative.raw_fitness: # representative is always the fittest network
            self.representative = network

    def calculate_total_adjusted_fitness(self):

        # calculate raw fitness for each network in species
        total_adjusted_fitness = 0
        for net in self.networks:
            result1 = net.compute_inputs(0,0)
            result2 = net.compute_inputs(0,1)
            result3 = net.compute_inputs(1,0)
            result4 = net.compute_inputs(1,1)

            error = (result1 - 0)**2 + (result2 - 1)**2 + (result3 - 1)**2 + (result4 - 0)**2
            net.raw_fitness = 1 / (1+error)
            net.adjusted_fitness = net.raw_fitness / len(self.networks)
            total_adjusted_fitness += net.adjusted_fitness


            # print("---- Fitness Calculation ----")
            # print("Result 1: ", result1)
            # print("Result 2: ", result2)
            # print("Result 3: ", result3)
            # print("Result 4: ", result4)
            # print("Error: ", error)
            # print("Raw Fitness: ", net.raw_fitness)
            # print("Adjusted Fitness: ", net.adjusted_fitness)

        self.total_adjusted_fitness = total_adjusted_fitness


    def remove_lowperforming_networks(self):
        if len(self.networks) > 1:
            surviving_networks_count = math.floor(len(self.networks) * (self.survival_rate / 100))
            raw_fitnesses = [network.raw_fitness for network in self.networks]
            surviving_networks = []

            if surviving_networks_count > 0:
                while (surviving_networks_count > 0):
                    highest_value = max(raw_fitnesses)
                    idx_of_highest_value = raw_fitnesses.index(highest_value)
                    surviving_networks.append(self.networks[idx_of_highest_value])
                    raw_fitnesses[idx_of_highest_value] = 0
                    surviving_networks_count -= 1
                
                self.networks = surviving_networks
                self.representative = surviving_networks[0]

    def reset_species(self):
        self.networks = []

    def calculate_compatibility_distance(self, network: Network):
        
        # get disjoint genes
        disjoint_genes = 0
        for gene in self.representative.connections:
            if gene.connection_id not in [g.connection_id for g in network.connections]:
                disjoint_genes += 1
        for gene in network.connections:
            if gene.connection_id not in [g.connection_id for g in self.representative.connections]:
                disjoint_genes += 1

        # if genes have different lengths, get amount of disjoint genes
        number_genes_larger_genome = None
        excess_genes = 0
        if len(network.connections) != len(self.representative.connections):
            if len(network.connections) > len(self.representative.connections):
                longer_genome = network
                short_genome = self.representative
            else:
                longer_genome = self.representative
                short_genome = network
            
            number_genes_larger_genome = len(longer_genome.connections)
            max_id_short_genome = 0
            for gene in short_genome.connections:
                if gene.connection_id > max_id_short_genome:
                    max_id_short_genome = gene.connection_id

            excess_genes = len([g for g in longer_genome.connections if g.connection_id > max_id_short_genome])
            disjoint_genes -= excess_genes
        else:
            number_genes_larger_genome = len(network.connections)

        # get matching genes and calculate their average weight difference
        matches = self.get_matching(self.representative, network)[1]
        sum = 0
        for conn1, conn2 in matches:
            sum += abs(conn1.weight - conn2.weight)
        average_weight_distance = sum / len(matches)
        compatibilty_dist = ((self.c1 * excess_genes) / number_genes_larger_genome) + ((self.c2 * disjoint_genes) / number_genes_larger_genome) + (self.c3 * average_weight_distance)
        return compatibilty_dist
    

    def produce_offspring(self,
                          children_amount,
                          mutation_offspring_rate,
                          weight_mutation_perturbation_rate,
                          weight_mutation_random_value_rate,
                          bias_weight_mutation_rate,
                          crossover_offspring_rate,
                          new_neuron_probability,
                          new_connection_probability,
                          hist_marker: HistoricalMarker,
                          elit_nets_amount
                          ):
        children = []

        # always keep the best performing networks in the species
        #print("Elits: ") 
        raw_fitnesses = [n.raw_fitness for n in self.networks]
        elit_networks = []
        for i in range(0, elit_nets_amount):
            if i < len(self.networks):
                max_raw_fitness = max(raw_fitnesses)
                idx_of_max = raw_fitnesses.index(max_raw_fitness)
                elit_networks.append(copy.deepcopy(self.networks[idx_of_max]))
                raw_fitnesses[idx_of_max] = 0
                #print(self.networks[i])
                children_amount -= 1

        children += elit_networks

        network_mutations_amount = round(children_amount * (mutation_offspring_rate / 100)) if len(self.networks) > 1 else children_amount
        crossover_amount = round(children_amount * (crossover_offspring_rate / 100)) if len(self.networks) > 1 else 0

        # print("----- Reproduction -----")
        # print("Species has ", str(len(self.networks)) + " networks")
        # print("Species has average adjusted fitness of ", self.average_adjusted_fitness)
        # print("Species calculated children amount:", children_amount)
        # print("Species mutations amount: ", network_mutations_amount)
        # print("Species crossovers amount: ", crossover_amount)

        for _ in range(0, network_mutations_amount):
            # create new networks using mutations
            child = self.mutate_network(weight_mutation_perturbation_rate, weight_mutation_random_value_rate, bias_weight_mutation_rate)
            if child != None:
                children.append(child)
    
        for _ in range(0, crossover_amount):
            # create new networks using crossovers
            child = self.crossover_networks()
            if child != None:
                children.append(child)
   
        if len(children) == 0:
            return []

        # add new neuron
        random_num = random.randint(0, 100)
        if (random_num <= new_neuron_probability):
            random_network = random.choice(children)
            random_network.mutate_with_new_neuron(hist_marker)

        # add new connection
        random_num = random.randint(0, 100)
        if (random_num <= new_connection_probability):
            random_network = random.choice(children)
            random_network.mutate_with_new_connection(hist_marker)

        return children, elit_networks


    # chooses random networks from species and mutates them
    def mutate_network(self, weight_mutation_perturbation_rate, weight_mutation_random_value_rate, bias_weight_mutation_rate):
        if len(self.networks) == 0:
            return None
        network = random.choice(self.networks)
        random_value = random.uniform(1, 100)

        if (random_value <= bias_weight_mutation_rate):
            # mutate bias weights of neuron in network
            network.apply_bias_weight_mutation()

        elif (random_value <= weight_mutation_random_value_rate):
            # mutate connection weights in network by setting new random values
            network.apply_random_weight_mutation()

        elif (random_value <= weight_mutation_perturbation_rate):
            # mutate connection weights in network by perturbation
            network.apply_weight_perturbation()
        
        return network
    
    # method to get list of matching genes from two genomes
    def get_matching(self, genome_1, genome_2):
        matching_rand_choice = []
        pairs = []
        for connection in genome_1.connections:
            for conn in genome_2.connections:
                pairs.append((connection, conn))
                if (connection.connection_id == conn.connection_id) and (connection.connection_id not in [c.connection_id for c in matching_rand_choice]):
                    matching_rand_choice.append(random.choice([connection, conn]))
        return (matching_rand_choice, pairs)

    # crossover random networks to create offspring
    def crossover_networks(self):
        if len(self.networks) < 2:
            return None
        
        rand_idx_1 = random.randint(0, len(self.networks)-1)
        rand_idx_2 = random.randint(0, len(self.networks)-1)
        
        while (rand_idx_1 == rand_idx_2):
            rand_idx_2 = random.randint(0, len(self.networks)-1)
        
        parent_1 = self.networks[rand_idx_1]
        parent_2 = self.networks[rand_idx_2]

        child_connections = []

        # find matching connections in both networks and choose random to inherit 
        child_connections += self.get_matching(parent_1, parent_2)[0]

        # if parent_1 is fitter, inherit all adjoint and excess connections in parent_1 to child      
        if parent_1.adjusted_fitness >= parent_2.adjusted_fitness:
            for connection in parent_1.connections:
                if (connection.connection_id not in [conn.connection_id for conn in parent_2.connections]) and (connection.connection_id not in [c.connection_id for c in child_connections]):
                    child_connections.append(connection)
        
        # if parent_2 is fitter, inherit all adjoint and excess connections in parent_2 to child
        else:
            for connection in parent_2.connections:
                if (connection.connection_id not in [conn.connection_id for conn in parent_1.connections]) and (connection.connection_id not in [c.connection_id for c in child_connections]):
                    child_connections.append(connection)

        crossover_network = Network(2,1)
        neurons = []

        # add all neurons used in connections to neuron list
        for connection in child_connections:
            crossover_network.add_existing_connection(connection)
            if connection.neuron_in.id not in [n.id for n in neurons]: neurons.append(connection.neuron_in)
            if connection.neuron_out.id not in [n.id for n in neurons]: neurons.append(connection.neuron_out)
        
        highest_neuron_id = max([neuron.id for neuron in neurons])

        # adjust neuron id count in network
        crossover_network.neuron_ids = highest_neuron_id + 1

        input_neurons = []
        output_neurons = []
        hidden_neurons = []
        
        for neuron in neurons:
            if isinstance(neuron, Input_neuron):
                input_neurons.append(neuron)
            elif isinstance(neuron, Output_neuron):
                output_neurons.append(neuron)
            else:
                hidden_neurons.append(neuron)
        
        crossover_network.input_neurons = input_neurons
        crossover_network.output_neurons = output_neurons
        crossover_network.hidden_neurons = hidden_neurons

        return crossover_network
    
    def __str__(self):
        return "[ Total adjusted fitness: " + str(self.total_adjusted_fitness) + " Network Amount: " + str(len(self.networks)) + " ]"