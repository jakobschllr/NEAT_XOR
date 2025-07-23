from neat_classes.PopulationHandler import PopulationHandler
from neat_classes.tests import test_mutations, test_network_forward_pass

def main():
    population_handler = PopulationHandler(150,2,1)
    population_handler.initial_population()
    population_handler.start_evolution_process() 
    

#run_tests()
main()