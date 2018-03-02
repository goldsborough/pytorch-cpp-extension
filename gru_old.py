class FunkyGRUFunction(Function):
    @staticmethod
    def forward(ctx, input, state, input_weights, state_weights, input_biases,
                state_biases):
        # weights = (3, features + state_size, state_size)
        # state -> (batch_size, state_size)
        # input -> (batch_size, features)

        # batch_size x 3 * state_size
        input_gates = F.linear(input, input_weights, input_biases)
        # batch_size x 3 * state_size
        state_gates = F.linear(state, state_weights, state_biases)

        # Split the gates into 3 x batch_size x state_size
        input_gates = input_gates.chunk(3, dim=1)
        state_gates = state_gates.chunk(3, dim=1)

        # # r_t = sigmoid(W_r [x_t, h_{t-1}] + b_r)
        reset_gate = F.sigmoid(input_gates[0] + state_gates[0])
        # z_t = sigmoid(W_z [x_t, h_{t-1}] + b_z)
        update_gate = F.sigmoid(input_gates[1] + state_gates[1])
        # # c_t = tanh(W_c * [x_t, r * h_{t-1}] + b_c)
        candidate_state = F.tanh(input_gates[2] + reset_gate * state_gates[2])

        # h_t = z_t * h_{t-1} + (1 - z_t) * c_t
        new_state = update_gate * state + (1 - update_gate) * candidate_state

        return new_state

    @staticmethod
    def backward(ctx, grad_output):
        pass


class FunkyGRU(nn.Module):
    def __init__(self, input_features, state_size):
        super(FunkyGRU, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.input_weights = nn.Parameter(
            torch.Tensor(3 * state_size, input_features))
        self.state_weights = nn.Parameter(
            torch.Tensor(3 * state_size, state_size))
        self.input_biases = nn.Parameter(torch.Tensor(3 * state_size))
        self.state_biases = nn.Parameter(torch.Tensor(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return FunkyGRUFunction.apply(input, state, self.input_weights,
                                      self.state_weights, self.input_biases,
                                      self.state_biases)
