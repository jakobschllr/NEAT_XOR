import math
import random

class Neuron():
    def __init__(self, id, bias):
        self.id = id
        self.current_weighted_inputs = []
        self.raw_output = 0
        self.bias_weight = random.uniform(-20.0, 20.0)
        self.bias = bias
        self.bias_perturbation_strength = 2.0

    def calculate_output(self, input, activation_function):
        if activation_function == "modified-sigmoid":
            clipped_input = max(input, -144)
            result = 1 / (1 + math.pow(math.e, - clipped_input)) #-4.9 *
            return result
        
    def get_weighted_bias(self):
        return self.bias * self.bias_weight
    
    def add_weighted_input(self, weighted_input):
        self.current_weighted_inputs.append(weighted_input)

    def mutate_bias_weight(self):
        #self.bias_weight = random.uniform(-1.0, 1.0) # set random new bias value
        random_value = random.uniform(0.0, 1.0)
        if random_value < 0.5:
            perturbation_value = self.bias_perturbation_strength
        else:
            perturbation_value = - self.bias_perturbation_strength
        self.bias_weight += perturbation_value

    def reset(self):
        self.current_weighted_inputs = []
        self.raw_output = 0

    def __str__(self):
        return str(self.id) + "|" + str(round(self.bias_weight, 3)) + " "