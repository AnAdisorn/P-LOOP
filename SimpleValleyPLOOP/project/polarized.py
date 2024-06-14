# %%
from grph_functions import graphene_bloch, p, c, h, a, delta
from scipy.integrate import odeint
from math import sqrt
from math import pi
import cmath
import numpy as np
import sys

# %% Get params
params = np.load("params.npy")

A0 = params[0] * delta
lamda = params[1]

# %% Take lamb input
N = 300  # Sets the iteration until which dynamics is done
lam = lamda * p  # Sets wavelength
freq = c / lam  # Defines frequency
freq = freq / h  # Converts frequency to atomic units
w = 2.0 * pi * freq  # Sets angular frequency
T_w = 2.0 * pi / w  # Cycle duration
T = 0.2 * T_w  # Pulse duration(Intensity FWHM)
ts = np.linspace(-0.5 * T_w, 0.5 * T_w, N)  # Defining the time array
print(T)

# %% Calculate upper valleys
num_grids = 250
P = np.zeros([num_grids, num_grids])
d = 4 * pi / (3 * a)
dp = np.linspace(0, d, num_grids)

P_band = np.zeros(2)
for band in range(2):
    for i in range(num_grids):
        print(f"{band=}, {i=}")
        q_x = dp[i]
        for j in range(num_grids):
            if band == 0:
                q_y = -dp[j]
            else:
                q_y = dp[j]
            vt = sqrt(3) * i
            ft = -1.0 * sqrt(3) * i + num_grids * sqrt(3)

            x = np.array([1.0, 0.0, 0.0])
            if j < min(vt, ft):
                xq = odeint(graphene_bloch, x, ts, args=(A0, q_x, q_y, w, T))  # solves the ode
                d1 = xq[:, 0]  # rho_vv

                P[i, j] = 1 - d1[-1]  # excitation

    P_band[band] = P.sum()

np.save("./result/output.npy", (P_band[1]-P_band[0])/(P_band[1]+P_band[0]))
