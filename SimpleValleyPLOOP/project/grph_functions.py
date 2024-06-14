import numpy as np
from math import sqrt
from cmath import phase
import numba as nb

# Constants for atimic unit
p = 10 ** (-7)
c = 3.0 * 10 ** (8)
h = 4.13 * 10 ** (16)

# Constants for graphene
gamma = 0.0024  # Dephasing rate with time of 10 femtoseconds
h = 4.13 * 10 ** (16)
a = 2.68  # Lattice spacing
g = 0.11  # Hopping
delta = 4.0 * np.pi / (3 * sqrt(3) * a)  # Separation in momentum space of Dirac points

@nb.njit()
def A_field(t, A0, w, T):
    # Vector potential with Gaussian envelope
    A = A0 * (np.exp(-1.38 * t * t / (T**2))) * (np.cos(w * t))
    return A

@nb.njit()
def E_field(t, A0, w, T):
    # Corresponding Electric field
    E = A0 * w * np.exp(-1.38 * (t * t) / (T**2)) * np.sin(w * t) + A0 * np.exp(
        -1.38 * (t * t) / (T**2)
    ) * (2 * 1.38 * t / (T**2)) * np.cos(w * t)
    return -E

@nb.njit()
def off_diagonal(q_x, q_y):
    return -g * (
        np.exp(1j * a * q_x)
        + 2 * np.exp(-1j * a * q_x / 2) * np.cos(sqrt(3) * a * q_y / 2)
    )

@nb.njit()
def dipole_element(q_x, q_y):
    d1x = 1 + 2 * np.cos(sqrt(3) * a * q_y)
    d1y = np.sin(3 * a * q_x / 2) * np.sin(sqrt(3.0) * a * q_y / 2)

    d2 = 2 * np.cos(sqrt(3) * a * q_y)
    d3 = 2 * np.cos(0.5 * a * (3 * q_x - sqrt(3) * q_y))
    d4 = 2 * np.cos(0.5 * a * (3 * q_x + sqrt(3) * q_y))

    dphi_dx = 0.25 * (a - 3 * a * d1x / (3 + d2 + d3 + d4))  # d(phase)/dx
    dip_x = -0.5 * dphi_dx

    dphi_dy = sqrt(3.0) * a * d1y / (3 + d2 + d3 + d4)  # d(phase)/dy
    dip_y = -0.5 * dphi_dy

    return dip_x, dip_y

@nb.njit()
def graphene_bloch(x, t, A0, q_x, q_y, w, T):
    # Variables are crystal momentum and the time

    # properly define the vector potential as the definition here
    k_y = q_y + A_field(t, A0, w, T)  # TODO: same apply to k_x?
    f = off_diagonal(q_x, k_y)
    # H = np.array([[0.0, f], [np.conj(f), 0.0]])  # Defining the Hamiltonian

    E1 = abs(f)  # conduction band energy
    E2 = -abs(f)  # valence band energy

    del_E = E1 - E2  # band gap

    # to determine the elements of the eigenstates( use form {1, +/- e^(i*phi)})
    # y1 = np.exp(1j * phi)
    # y2 = np.exp(-1j * phi)

    # define the conduction and valence band eigenvectors:
    # C = (np.array([1, y2])) / sqrt(2.0)
    # V = (np.array([1, -1 * y2])) / sqrt(2.0)

    # define the dipole matrix elements (as in the definition:)
    dip_x, dip_y = dipole_element(
        q_x, k_y  # TODO: same apply to k_x?
    )  # calculating the dynamic dipole elements
    # assigning each ODE variable to the vector x:
    c1, c3, c4 = x

    # defining the real and imaginary parts of density matrix elements

    p1 = c1  # on-diagonal term for the valence band
    p2 = c3 + 1j * c4  # off- diagonal term (rho_cv)

    # defining the eqns (hopefully works !!) we just consider E-y for this example
    # TODO: E*dip_y --> E.dot(dipole_moment)
    rho_vv = 1.0j * (E_field(t, A0, w, T)) * (dip_y) * p2 + np.conj(
        1.0j * (E_field(t, A0, w, T)) * (dip_y) * p2
    )
    dc1dt = rho_vv.real  # valence band population
    # dc2dt = rho_vv.imag
    rho_cv = -(1j * (del_E) + gamma) * p2 + 1j * E_field(t, A0, w, T) * (dip_y) * (
        2 * p1 - 1
    )
    dc3dt = rho_cv.real
    dc4dt = rho_cv.imag
    return [dc1dt, dc3dt, dc4dt]
