from neat_classes.network import Network
from neat_classes.PopulationHandler import PopulationHandler
from neat_classes.tests import test_mutations, test_network_forward_pass, simple_network, sigmoid
import random

def main():
    population_handler = PopulationHandler()
    population_handler.initial_population()
    population_handler.start_evolution_process() 

def run_tests():
    #test_mutations()
    simple_network()
    print("-----------------------")
    test_network_forward_pass()
    

#run_tests()
main()

# noch ergänzen:
# interspecies mating
# echte Zyklensicherheit, im Moment wird in Network nur geprüft ob keine direkten zyklen existieren,
# aber auch indirekte Zyklen müssen verhindert weden
