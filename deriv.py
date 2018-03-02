import numpy as np

# (x + s(y)) / (s(x) + (x + y)^2)

x = 3
y = -4

f = (x + (1 / (1 + np.exp(-y)))) / ((1 / (1 + np.exp(-x))) + (x + y)**2)
print(f)

sigy = 1 / (1 + np.exp(-y))
num = x + sigy
sigx = 1 / (1 + np.exp(-x))
xpy = x + y
xpysq = xpy**2
den = sigx + xpysq
invden = 1.0 / den
f = num * invden

dnum = invden
dinvden = num
dden = dinvden * (-1.0 / (den**2))
dsigx = dden * (1)
dxpysq = dden * (1)
dxpy = dxpysq * 2 * xpy
dx = dxpy * (1)
dy = dxpy * (1)
dx += dsigx * (1 - sigx) * sigx
dx += dnum * (1)
dsigy = dnum * (1)
dy += dsigy * (1 - sigy) * sigy
