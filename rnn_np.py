import numpy as np

np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    s = sigmoid(z)
    return (1 - s) * s


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def d_tanh(z):
    t = tanh(z)
    return 1 - (t * t)


class RNN(object):
    def __init__(self, input_features, state_size):
        self.W = np.random.randn(input_features + state_size, state_size)
        self.b = np.zeros(state_size)
        self.weights = [self.W, self.b]
        self.cache = {}

    def forward(self, input, previous_h):
        X = np.concatenate([input, previous_h], axis=1)
        z = X.dot(self.W) + self.b
        h = tanh(z)

        self.cache['z'] = z
        self.cache['X'] = X

        return h.sum()

    def backward(self, d_output):
        d_z = d_tanh(self.cache['z']) * d_output
        d_b = d_z
        d_W = self.cache['X'].T.dot(d_z)

        return [d_W, d_b]


class LSTM(object):
    def __init__(self, input_features, state_size):
        self.W = np.random.randn(input_features + state_size, 4 * state_size)
        self.b = np.zeros([1, 4 * state_size])
        self.weights = [self.W, self.b]
        self.cache = {}

    def forward(self, input, old_cell, old_h):
        X = np.concatenate([old_h, input], axis=1)

        gates = X.dot(self.W) + self.b
        gates = np.split(gates, 4, axis=1)

        input_gate = sigmoid(gates[0])
        forget_gate = sigmoid(gates[1])
        output_gate = sigmoid(gates[2])
        candidate_cell = tanh(gates[3])

        new_cell = old_cell * forget_gate + candidate_cell * input_gate
        new_h = tanh(new_cell) * output_gate

        self.cache['X'] = X
        self.cache['input_gate'] = input_gate
        self.cache['output_gate'] = output_gate
        self.cache['old_cell'] = old_cell
        self.cache['new_cell'] = new_cell
        self.cache['candidate_cell'] = candidate_cell
        self.cache['gates'] = gates

        return new_h.sum()

    def backward(self, d_output):
        d_output_gate = tanh(self.cache['new_cell']) * d_output
        d_tanh_new_cell = self.cache['output_gate'] * d_output
        d_new_cell = d_tanh(self.cache['new_cell']) * d_tanh_new_cell

        d_forget_gate = self.cache['old_cell'] * d_new_cell
        d_candidate_cell = self.cache['input_gate'] * d_new_cell
        d_input_gate = self.cache['candidate_cell'] * d_new_cell

        d_input_gate *= d_sigmoid(self.cache['gates'][0])
        d_forget_gate *= d_sigmoid(self.cache['gates'][1])
        d_output_gate *= d_sigmoid(self.cache['gates'][2])
        d_candidate_cell *= d_tanh(self.cache['gates'][3])

        d_gates = np.concatenate(
            [d_input_gate, d_forget_gate, d_output_gate, d_candidate_cell],
            axis=1)

        d_X = self.W.dot(d_gates.T)
        d_input = np.split(d_X, [d_input_gate.size], axis=0)[1]
        d_W = self.cache['X'].T.dot(d_gates)
        d_b = d_gates

        return [d_input, d_W, d_b]


class GRU(object):
    def __init__(self, input_features, state_size):
        self.W_input = np.random.randn(input_features, 3 * state_size)
        self.W_state = np.random.randn(state_size, 3 * state_size)
        self.b_input = np.zeros([1, 3 * state_size])
        self.b_state = np.zeros([1, 3 * state_size])
        self.weights = [self.W_input, self.W_state, self.b_input, self.b_state]
        self.cache = {}

    def forward(self, input, old_h):
        input_gates = input.dot(self.W_input) + self.b_input
        state_gates = old_h.dot(self.W_state) + self.b_state

        input_gates = np.split(input_gates, 3, axis=1)
        state_gates = np.split(state_gates, 3, axis=1)

        z_reset = input_gates[0] + state_gates[0]
        reset_gate = sigmoid(z_reset)

        z_update = input_gates[1] + state_gates[1]
        update_gate = sigmoid(z_update)

        z_candidate = input_gates[2] + reset_gate * state_gates[2]
        candidate_h = tanh(z_candidate)

        # new_h = update_gate * candidate_h + (1 - update_gate) * old_h
        new_h = update_gate * (candidate_h - old_h) + old_h

        self.cache['input'] = input
        self.cache['old_h'] = old_h
        self.cache['state_gates'] = state_gates
        self.cache['input_gates'] = input_gates
        self.cache['reset_gate'] = reset_gate
        self.cache['update_gate'] = update_gate
        self.cache['z_candidate'] = z_candidate
        self.cache['z_update'] = z_update
        self.cache['z_reset'] = z_reset
        self.cache['candidate_h'] = candidate_h

        return new_h.sum()

    def backward(self, d_output):
        d_candidate_h = self.cache['update_gate'] * d_output
        d_update_gate = self.cache['candidate_h'] - self.cache['old_h']
        d_update_gate *= d_output

        d_state_gates = np.zeros([3, self.W_state.shape[0]])
        d_input_gates = np.zeros_like(d_state_gates)

        d_z_candidate = d_tanh(self.cache['z_candidate']) * d_candidate_h
        d_reset_gate = self.cache['state_gates'][2] * d_z_candidate
        d_state_gates[2] = self.cache['reset_gate'] * d_z_candidate
        d_input_gates[2] = d_z_candidate

        d_update_gate *= d_sigmoid(self.cache['z_update'])
        d_input_gates[1] = d_update_gate
        d_state_gates[1] = d_update_gate

        d_reset_gate *= d_sigmoid(self.cache['z_reset'])
        d_input_gates[0] = d_reset_gate
        d_state_gates[0] = d_reset_gate

        d_input_gates = d_input_gates.reshape(1, -1)
        d_state_gates = d_state_gates.reshape(1, -1)

        d_W_input = self.cache['input'].T.dot(d_input_gates)
        d_b_input = d_input_gates

        d_W_state = self.cache['old_h'].T.dot(d_state_gates)
        d_b_state = d_state_gates

        return [d_W_input, d_W_state, d_b_input, d_b_state]


def check_weights(rnn, inputs, D):
    for w, W in enumerate(rnn.weights):
        for i in range(W.size):
            w_0 = W.flat[i]
            W.flat[i] = w_0 - D
            left = rnn.forward(*inputs)
            W.flat[i] = w_0 + D
            right = rnn.forward(*inputs)
            W.flat[i] = w_0
            num = (right - left) / (2 * D)
            # print(num.flatten())

            rnn.forward(*inputs)
            grad = rnn.backward(np.ones_like(num))
            # print(grad[w].flatten())
            grad = grad[1 + w].flat[i]

            error = np.abs(num - grad) / np.maximum(np.abs(num), np.abs(grad))
            if error > 1e-3:
                print('!!! {}[{}] {:e} | {:.8f} vs. {:.8f}'.format(
                    w, i, error, num, grad))
                # assert False
            elif error > 1e-4:
                print('{}[{}] {:e} | {:.8f} vs. {:.8f}'.format(
                    w, i, error, num, grad))


def check_input(rnn, inputs, D):
    X = inputs[0]
    for i in range(X.size):
        x_0 = X.flat[i]
        X.flat[i] = x_0 - D
        left = rnn.forward(*inputs)
        X.flat[i] = x_0 + D
        right = rnn.forward(*inputs)
        X.flat[i] = x_0
        num = (right - left) / (2 * D)

        rnn.forward(*inputs)
        grad = rnn.backward(np.ones_like(num))
        grad = grad[0].flat[i]

        error = np.abs(num - grad) / np.maximum(np.abs(num), np.abs(grad))
        if error > 1e-3:
            print('!!! [{}] {:e} | {:.6f} vs. {:.6f}'.format(
                i, error, num, grad))
            # assert False
        elif error > 1e-5:
            print('[{}] {:e} | {:.6f} vs. {:.6f}'.format(i, error, num, grad))


def main():
    B = 1
    I = 10
    S = 4
    D = np.float64(1e-6)

    X = np.random.randn(B, I)
    C = np.random.randn(1, S)
    h = np.random.randn(1, S)
    inputs = [X, h, C]
    # inputs = [X, h]

    rnn = LSTM(I, S)

    # check_weights(rnn, inputs, D)
    check_input(rnn, inputs, D)

    print('Ok')

if __name__ == '__main__':
    main()
