#include <torch/torch.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
at::Tensor d_sigmoid(at::Tensor z) {
  auto s = at::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
at::Tensor d_tanh(at::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
at::Tensor d_elu(at::Tensor z, at::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<at::Tensor> lltm_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  auto X = at::cat({old_h, input}, /*dim=*/1);

  auto gates = at::addmm(bias, X, weights.transpose(0, 1));

  const auto state_size = old_h.size(1);
  auto input_gate = at::sigmoid(gates.slice(/*dim=*/1, 0, state_size));
  auto output_gate =
      at::sigmoid(gates.slice(/*dim=*/1, state_size, 2 * state_size));
  auto candidate_cell =
      at::elu(gates.slice(/*dim=*/1, 2 * state_size), /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = at::tanh(new_cell) * output_gate;

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}

std::vector<at::Tensor> lltm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gates,
    at::Tensor weights,
    at::Tensor old_cell) {
      auto d_gates = at::zeros_like(gates);
      const auto state_size = grad_h.size(1);

  // d_gates.slice(1, state_size, 2 * state_size) = at::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;


  auto d_old_cell = d_new_cell;
  // d_gates.slice(1, 2 * state_size) = input_gate * d_new_cell;
  // d_gates.slice(1, 0, state_size) = candidate_cell * d_new_cell;
  //
  // d_gates.slice(1, 0, 2 * state_size) *= d_sigmoid(gates.slice(/*dim=*/1, 0, 2 * state_size));
  // // d_output_gate *=
  // //     d_sigmoid(gates.slice(/*dim=*/1, state_size, 2 * state_size));
  // d_gates.slice(1, 2 * state_size) *= d_elu(gates.slice(/*dim=*/1, 2 * state_size));

  // auto d_gates =
  //     at::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights).split(state_size, /*dim=*/1);

  return {d_X[0], d_X[1], d_weights, d_bias, d_old_cell};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
