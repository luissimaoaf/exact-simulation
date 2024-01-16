import numpy as np
import sampling
from scipy.optimize import minimize_scalar
from scipy import integrate
import time
import matplotlib.pyplot as plt
import sys
import csv

lamb0 = 0.0110
lamb1 = 0.1000

min_jump = 0.0113
max_jump = 0.0312

X_0 = 0.0422
mean = 0.0422
kappa = 0.0117
sigma = 0.0130

T = 3
T_max = 3
strike = 0.05
periods = 3
cap_limits = [1, 2, 3]

# model functions


# lamperti transform
def f(x):
    return 2*(np.sqrt(x) - np.sqrt(X_0))/sigma


# inverse lamperti transform
def f_inv(x):
    return (sigma*x/2 + np.sqrt(X_0))**2


# drift function
def mu_y(x):
    return (4*kappa*mean - sigma**2)/(2*sigma**2)/(x+2*np.sqrt(X_0)/sigma) - kappa/2 * (x+2*np.sqrt(X_0)/sigma)


def jump_int(x):
    return lamb0 + lamb1 * x


def delta_y(x, z):
    return 2 * (np.sqrt((np.sqrt(X_0) + x*sigma/2)**2 + z) - np.sqrt(X_0))/sigma - x


def phi_y(x):
    return 0.5 * (-(4*kappa*mean - sigma**2)/(2*sigma**2)/(x+2*np.sqrt(X_0)/sigma)**2
                  - kappa/2 + mu_y(x)**2)


# function A(y) needed for the acceptance tests
def integrated_drift(x):
    return (4*kappa*mean - sigma**2)/(2*sigma**2) * np.log(1 + x*sigma/(2*np.sqrt(X_0))) \
           - kappa/2 * (x**2/2 + 2*np.sqrt(X_0)*x/sigma)


def h(t, precision=1e-10, max_iter=100):

    h_value = 1 / np.sqrt(2 * np.pi * t ** 3) * np.exp(-1 / (2 * t))
    j = 1

    p_term = (2 * j + 1) * np.exp(-(2 * j + 1) ** 2 / (2 * t))
    n_term = (2 * j - 1) * np.exp(-(2 * j - 1) ** 2 / (2 * t))
    summand = (-1) ** j / np.sqrt(2 * np.pi * t ** 3) * (p_term - n_term)

    h_value += summand

    while abs(summand) > precision and j < max_iter:

        p_term = (2 * j + 1) * np.exp(-(2 * j + 1) ** 2 / (2 * t))
        n_term = (2 * j - 1) * np.exp(-(2 * j - 1) ** 2 / (2 * t))
        summand = (-1) ** j / np.sqrt(2 * np.pi * t ** 3) * (p_term - n_term)
        h_value += summand

        j += 1

    return h_value


def objective_function2(y, theta):
    phi_max = -minimize_scalar(lambda x: -phi_y(x), bounds=(y - theta, y + theta), method='bounded').fun
    phi_min = minimize_scalar(phi_y, bounds=(y - theta, y + theta), method='bounded').fun
    c = -minimize_scalar(lambda x: -jump_int(f_inv(x)), bounds=(y - theta, y + theta), method='bounded').fun
    K = -minimize_scalar(lambda x: -np.exp(integrated_drift(x)), bounds=(y - theta, y + theta), method='bounded').fun
    S = max(np.exp(-phi_min*T), 1)

    val1 = integrate.quad(lambda t: h(t/theta**2), 0, T)[0]
    val2 = integrate.quad(lambda t: t*h(t/theta**2)*np.exp(-phi_max*t), 0, T)[0]
    val3 = integrate.quad(lambda t: t*h(t/theta**2)*np.exp(-(phi_max+c)*t), 0, T)[0]
    val4 = integrate.quad(lambda t: h(t/theta**2)*np.exp(-(phi_max+c)*t), 0, T)[0]

    term1 = c*np.exp(integrated_drift(y))*(val1 - (phi_max+c)*val3 - val4)/(K*S*theta**2*(phi_max+c)**2)
    term2 = (np.exp(integrated_drift(y+theta)) + np.exp(integrated_drift(y-theta)))*val2/(2*K*S*theta**2)
    term3 = c*np.exp(integrated_drift(y))*(1-val1/theta**2)*((1-np.exp(-(phi_max+c)*T))/(phi_max+c)
                                                             + (T*np.exp(-(phi_max+c)*T))/c)/(K*S)

    return -(term1+term2+term3)


# res = minimize_scalar(lambda x: objective_function2(-3.5, x), bounds=(0.05, 0.5), method='Bounded')
# print(res.x)
# print(res.fun)
#
# x = np.linspace(0.05, 10, 100)
# plt.plot(x, [-objective_function2(-2, t) for t in x])
# plt.plot(x, [-objective_function2(0, t) for t in x])
# plt.plot(x, [-objective_function2(20, t) for t in x])
# plt.show()

def main():
    theta_values = []
    y_range = np.linspace(-29, 40, 100)
    for y in y_range:
        if y < -20:
            u_bound = 30 + y
        else:
            u_bound = 10
        theta = minimize_scalar(lambda x: objective_function2(y, x), bounds=(0.005, u_bound), method='Bounded').x
        theta_values.append(theta)
        print(theta)
    with open('ajd_thetas.txt', 'w') as text_file:
        for theta in theta_values:
            text_file.write("%s\n" % theta)


main()


