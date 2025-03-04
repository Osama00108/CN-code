def sigmoid(x):
    return 1 / (1 + 2.718281828459045 ** (-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

w1, w2, w3, w4 = 0.3, 0.2, 0.3, 0.45
w5, w6, w7, w8 = 0.5, 0.4, 0.2, 0.5
b1, b2 = 0.5, 0.7

i1, i2 = 1, 1

target_o1, target_o2 = 0.1, 0.99 
learning_rate = 0.1  

# Forward pass
net_h1 = w1 * i1 + w2 * i2 + b1
net_h2 = w3 * i1 + w4 * i2 + b1

out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

net_o1 = w5 * out_h1 + w6 * out_h2 + b2
net_o2 = w7 * out_h1 + w8 * out_h2 + b2

out_o1 = sigmoid(net_o1)
out_o2 = sigmoid(net_o2)

error_o1 = 0.5 * (target_o1 - out_o1) ** 2
error_o2 = 0.5 * (target_o2 - out_o2) ** 2

total_error = error_o1 + error_o2

# Backpropagation
delta_o1 = (out_o1 - target_o1) * sigmoid_derivative(net_o1)
delta_o2 = (out_o2 - target_o2) * sigmoid_derivative(net_o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(net_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(net_h2)

w1 -= learning_rate * delta_h1 * i1
w2 -= learning_rate * delta_h1 * i2
w3 -= learning_rate * delta_h2 * i1
w4 -= learning_rate * delta_h2 * i2
w5 -= learning_rate * delta_o1 * out_h1
w6 -= learning_rate * delta_o1 * out_h2
w7 -= learning_rate * delta_o2 * out_h1
w8 -= learning_rate * delta_o2 * out_h2

b1 -= learning_rate * (delta_h1 + delta_h2)
b2 -= learning_rate * (delta_o1 + delta_o2)

net_h1_new = w1 * i1 + w2 * i2 + b1
net_h2_new = w3 * i1 + w4 * i2 + b1

out_h1_new = sigmoid(net_h1_new)
out_h2_new = sigmoid(net_h2_new)

net_o1_new = w5 * out_h1_new + w6 * out_h2_new + b2
net_o2_new = w7 * out_h1_new + w8 * out_h2_new + b2

out_o1_new = sigmoid(net_o1_new)
out_o2_new = sigmoid(net_o2_new)

print("Output before backpropagation:", out_o1, out_o2)
print("Total Error:", total_error)
print("Updated weights:", w1, w2, w3, w4, w5, w6, w7, w8)
print("Updated biases:", b1, b2)
print("Output after backpropagation:", out_o1_new, out_o2_new)
