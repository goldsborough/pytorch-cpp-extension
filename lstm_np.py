from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dsigmoid(z):
    return (1 - sigmoid(z)) * sigmoid(z)


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def dtanh(z):
    t = tanh(z)
    return 1 - (t * t)


class LSTM(object):
    def __init__(self, input_features, state_size):
        self.W = np.random.randn(input_features + state_size, 4 * state_size)
        self.b = np.zeros(4 * state_size)

        self.cache = {}
        self.gradients = {}

    def forward(self, input, hidden_state, cell_state):
        X = np.concatenate([input, hidden_state], axis=1)

        gates = X.dot(self.W) + self.b
        gates = np.split(gates, 4, axis=1)

        input_gate = sigmoid(gates[0])
        forget_gate = sigmoid(gates[1])
        output_gate = sigmoid(gates[2])
        candidate_cell = tanh(gates[3])

        new_cell_state = cell_state * forget_gate
        new_cell_state += candidate_cell * input_gate

        new_hidden_state = tanh(new_cell_state) * output_gate

        self.cache['h'] = hidden_state
        self.cache['C'] = cell_state
        self.cache['g_i'] = input_gate
        self.cache['g_f'] = forget_gate
        self.cache['g_o'] = output_gate
        self.cache['c_new'] = candidate_cell
        self.cache['gates'] = gates

        return new_hidden_state

    def backward(self, input, hidden_state, cell_state, grad_wrt_output):
        d_g_o = tanh(self.cache['C']) * grad_wrt_output[0]
        d_tanh_c = self.cache['g_o'] * grad_wrt_output[0]
        d_new_c = dtanh(cell_state) * d_tanh_c + grad_wrt_output[1]

        d_c_old = self.cache['g_f'] * d_new_c
        d_g_f = self.cache['C'] * d_new_c
        d_cand = self.cache['g_i'] * d_new_c
        d_g_i = self.cache['c_new'] * d_new_c

        d_g_c = dtanh(self.cache['gates'][3]) * d_cand
        d_g_o = dsigmoid(self.cache['gates'][2]) * d_g_o
        d_g_f = dsigmoid(self.cache['gates'][1]) * d_g_f
        d_g_i = dsigmoid(self.cache['gates'][0]) * d_g_i

        d_gates = np.concatenate([d_g_i, d_g_f, d_g_o, d_g_c], axis=1)

        X = np.concatenate([input, self.cache['h']], axis=1)

        self.gradients['b'] = d_gates
        self.gradients['W'] = X.T.dot(d_gates)

        d_X = d_gates.dot(self.W.T)
        d_input, d_h = np.split(d_X, [input.shape[1]], axis=1)

        # return d_input, d_c_old, d_h
        return d_h


from grad_check import gradient_check

B = 3
I = 4
S = 8

lstm = LSTM(I, S)

input = np.ones([1, I])
hidden_state = np.random.randn(1, S)
cell_state = np.random.randn(1, S)


def backward(*inputs):
    lstm.forward(*inputs)
    return lstm.backward(*inputs, grad_wrt_output=[np.ones(S), np.zeros(S)])


grad_wrt_output = [np.ones(S), np.zeros(S)]

lstm.forward(input, hidden_state, cell_state)
print(lstm.backward(input, hidden_state, cell_state, grad_wrt_output))

delta = 1e-8

left = lstm.forward(input, hidden_state, cell_state)
input[0] += delta
right = lstm.forward(input, hidden_state, cell_state)

print((right - left) / delta)

# gradient_check(
#     lstm.forward, backward, [input], extra=[hidden_state, cell_state])
