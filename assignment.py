import math

def tanh(x):
    return (2 / (1 + math.exp(-2 * x))) - 1

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def neural_network(inputs, weights1, weights2, bias1, bias2):
    
    hidden_inputs = [dot_product(inputs, w) + bias1 for w in weights1]
    hidden_outputs = [tanh(x) for x in hidden_inputs]
    
    output_inputs = [dot_product(hidden_outputs, w) + bias2 for w in weights2]
    final_outputs = [tanh(x) for x in output_inputs]
    
    return final_outputs

w1, w2, w3, w4 = 0.1, 0.2, 0.3, 0.4 
w5, w6, w7, w8 = 0.5, 0.2, 0.35, 0.25  

bias1, bias2 = 0.5, 0.7 

inputs = [1, 1]

weights1 = [[w1, w2], [w3, w4]]  
weights2 = [[w5, w6], [w7, w8]]  

outputs = neural_network(inputs, weights1, weights2, bias1, bias2)
print("Output of the network:", outputs[0], outputs[1])