import torch
from torch import nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
import math
import numpy as np

torch.manual_seed(42)


def d_sigmoid(z):
    s = F.sigmoid(z)
    return (1 - s) * s


def d_tanh(z):
    t = F.tanh(z)
    return 1 - (t * t)


class FunkyLSTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        X = torch.cat([old_h, input], dim=1)

        gate_weights = F.linear(X, weights, bias)
        gates = gate_weights.chunk(4, dim=1)

        input_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1])
        output_gate = F.sigmoid(gates[2])
        candidate_cell = F.tanh(gates[3])

        new_cell = old_cell * forget_gate + candidate_cell * input_gate
        new_h = F.tanh(new_cell) * output_gate

        ctx.save_for_backward(X, weights, forget_gate, input_gate, output_gate,
                              old_cell, new_cell, candidate_cell, gate_weights)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        X, weights, forget_gate = ctx.saved_variables[:3]
        input_gate, output_gate, old_cell = ctx.saved_variables[3:6]
        new_cell, candidate_cell, gate_weights = ctx.saved_variables[6:]

        d_output_gate = F.tanh(new_cell) * grad_h
        d_tanh_new_cell = output_gate * grad_h
        d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell

        d_old_cell = forget_gate * d_new_cell
        d_forget_gate = old_cell * d_new_cell
        d_candidate_cell = input_gate * d_new_cell
        d_input_gate = candidate_cell * d_new_cell

        gates = gate_weights.chunk(4, dim=1)
        d_input_gate *= d_sigmoid(gates[0])
        d_forget_gate *= d_sigmoid(gates[1])
        d_output_gate *= d_sigmoid(gates[2])
        d_candidate_cell *= d_tanh(gates[3])

        d_gates = torch.cat(
            [d_input_gate, d_forget_gate, d_output_gate, d_candidate_cell],
            dim=1)

        d_X = d_gates.mm(weights)
        d_old_h, d_input = d_X.split(d_input_gate.numel(), dim=1)
        d_weights = d_gates.t().mm(X)
        d_bias = d_gates

        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class FunkyLSTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(FunkyLSTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(4 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(4 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return FunkyLSTMFunction.apply(input, self.weights, self.bias, *state)


I = 2
S = 2

X = Variable(torch.randn(1, I).double(), requires_grad=True)
W = Variable(torch.randn(4 * S, I + S).double(), requires_grad=True)
b = Variable(torch.randn(1, 4 * S).double(), requires_grad=True)
h = Variable(torch.randn(1, S).double(), requires_grad=True)
C = Variable(torch.randn(1, S).double(), requires_grad=True)
# inputs = [X, W, b]

rnn = FunkyLSTM(I, S).type(torch.DoubleTensor)
# x = rnn(input[0], (h, C))
# print(x)
# x[0].sum().backward()

from torch.autograd import gradcheck

print(gradcheck(FunkyLSTMFunction.apply, [X, W, b, h, C]))
