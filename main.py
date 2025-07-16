from neat_classes.network import Network
from neat_classes.PopulationHandler import PopulationHandler
from neat_classes.test_mutations import test_mutations
import random

def main():
    population_handler = PopulationHandler()
    population_handler.initial_population()
    population_handler.start_evolution_process() 

def run_tests():
    test_mutations()

#run_tests()
main()

# noch ergänzen:
# interspecies mating
# 75% percent chance inherited gene is disabled if it is disabled in either parent