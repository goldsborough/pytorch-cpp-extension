import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dsigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def matmul_sum(A, B):
    return A.dot(B).sum()


def dmatmul_sum(A, B):
    D = np.ones([A.shape[0], B.shape[1]])
    dB = A.T.dot(D)
    dA = D.dot(B.T)
    return dA, dB


def gradient_check(forward,
                   backward,
                   inputs,
                   extra=None,
                   number_of_checks=100,
                   delta=1e-6,
                   tolerance=1e-5):
    inputs = np.array(inputs)
    extra = extra or []
    analytical = np.array(backward(*inputs, *extra))
    if not isinstance(analytical, list):
        analytical = np.expand_dims(analytical, axis=0)
    analytical = [i.flatten() for i in analytical]

    def scalar_forward(*args, **kwargs):
        return forward(*args, **kwargs).sum().squeeze().astype(np.float64)

    for n, input in enumerate(inputs):
        for _ in range(number_of_checks):
            i, = np.random.randint(input.size, size=1)
            element = input.flat[i]

            input.flat[i] = element - delta
            left = scalar_forward(*inputs, *extra)

            input.flat[i] = element + delta
            right = scalar_forward(*inputs, *extra).squeeze()

            numerical = (right - left) / (2 * delta)

            input.flat[i] = element

            analytical_here = analytical[n].flat[i]
            assert np.isscalar(analytical_here)

            error = np.abs(numerical - analytical_here)
            maximum = np.maximum(np.abs(numerical), np.abs(analytical_here))
            relative_error = error / maximum
            if relative_error > tolerance:
                print('Error @ input {}: {}'.format(n, relative_error))
                return False

    print('Ok')
    return True


def main():
    x = np.zeros([2])
    gradient_check(sigmoid, dsigmoid, [x])


if __name__ == '__main__':
    main()
