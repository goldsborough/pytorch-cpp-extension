import torch
from torch.autograd import Variable
from torch import nn

torch.manual_seed(42)

I = 4
S = 5

rnn = nn.LSTMCell(I, S)

input = Variable(torch.randn(1, 1, I))
h = Variable(torch.randn(1, S))
C = Variable(torch.randn(1, S))

x = rnn(input[0], (h, C))
# print(x)
