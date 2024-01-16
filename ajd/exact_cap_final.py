import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
import sampling

import time


# params

lamb0 = 0.0110
lamb1 = 0.1000

min_jump = 0.0113
max_jump = 0.0312

X_0 = 0.0422
mean = 0.0422
kappa = 0.0117
sigma = 0.0130

T = 3
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


def monte_carlo(n_sim=1000, sim_frames=[1000]):

    sim_count = 0
    sim_data = []
    cap_results = []
    optimal_thetas = [float(theta.strip()) for theta in open("ajd_thetas.txt", 'r')]

    # we start the timer
    start_time = time.time()

    # main simulation loop
    while sim_count < n_sim:
        print(sim_count)
        y = 0
        t = 0
        g = 1
        cap_value = 0
        cap_counter = 0

        # loop until we reach our horizon T
        while t < T:
            # choice of theta
            y_range = np.linspace(-29, 40, 100)
            if -29 < y < 40:
                for i in range(len(y_range)):
                    if y_range[i] < y:
                        theta_opt = (optimal_thetas[i] * (y_range[i + 1] - y) + optimal_thetas[i + 1] * (
                                y - y_range[i])) \
                                    / (y_range[i + 1] - y_range[i])
                        theta = min(theta_opt, (y - f(0)) / 4)
                        break
            else:
                theta = min(3, (y - f(0)) / 4)

            # determination of the minimum and maximum of phi
            # phi_max = -minimize_scalar(lambda x: -phi_y(x), bounds=(y - theta, y + theta), method='bounded').fun
            # phi_min = minimize_scalar(phi_y, bounds=(y - theta, y + theta), method='bounded').fun

            theta_min = -2.9081943013559806
            if y - theta < theta_min < y + theta:
                phi_min = -0.006002
                phi_max = max(phi_y(y - theta), phi_y(y + theta))
            elif y + theta < theta_min:
                phi_max = phi_y(y - theta)
                phi_min = phi_y(y + theta)
            elif theta_min < y - theta:
                phi_max = phi_y(y + theta)
                phi_min = phi_y(y - theta)

            lamb = jump_int(f_inv(y + theta))

            # bounds for sampling of estimator
            x_up = f_inv(y + theta)
            x_down = f_inv(y - theta)

            # main A/R scheme loop
            rejected = True
            while rejected:

                # generate the exit time for the interval [y-theta, y+theta]
                tau = theta ** 2 * sampling.generate_exit_time()

                # generate the required Poisson jump times
                candidate_times = sampling.generate_poisson_jumps(lamb, min(tau, T - t))
                test_times = sampling.generate_poisson_jumps(phi_max - phi_min, min(tau, T - t))
                exponential_times = sampling.generate_poisson_jumps(x_up - x_down, min(tau, T - t))

                # sort all times and add T-t if needed
                all_times = candidate_times + test_times + exponential_times

                if T - t < tau:
                    all_times.append(T - t)
                # we add the necessary cap checkpoints
                cap_times = [t_cap - t for t_cap in cap_limits if t < t_cap < min(T, t + tau)]
                all_times = all_times + cap_times
                all_times.append(tau)
                all_times = sorted(all_times)

                # generate a Brownian meander with exit time tau at all the required times and we rescale
                candidate_bridge_values = sampling.generate_brownian_meander([i / theta ** 2 for i in all_times])
                candidate_bridge_values = [x * theta for x in candidate_bridge_values]

                candidate_bridge = {all_times[i]: candidate_bridge_values[i] for i in range(len(all_times))}

                i = 0
                a = len(candidate_times)

                while i < a:
                    u = np.random.rand()
                    w = candidate_bridge[candidate_times[i]]

                    if u * lamb < jump_int(f_inv(y + w)):
                        break
                    else:
                        i += 1

                if i == a:

                    # no jump time was accepted
                    stopping_time = min(t + tau, T)

                    u, v = np.random.rand(2)

                    # first Bernoulli variable
                    # K = -minimize_scalar(lambda x: - integrated_drift(x),
                    #                      bounds=(y-theta, y+theta), method='bounded').fun

                    # previously computed location of global maximum
                    K_max = 1.382436647017447

                    if y-theta < K_max < y+theta:
                        K = 0.011016
                    else:
                        K = max(integrated_drift(y-theta), integrated_drift(y+theta))

                    w = candidate_bridge[min(tau, T - t)]
                    test_factor_1 = (np.log(u) < integrated_drift(y + w) - K)

                    # second Bernoulli variable
                    S = max(np.exp(-phi_min * (T - t)), 1)
                    test_factor_2 = (v * S < np.exp(-phi_min * min(tau, T - t)))

                    # thinning test process
                    test_factor_3 = True
                    j = 0
                    while j < len(test_times):

                        u = np.random.rand()
                        w = candidate_bridge[test_times[j]]
                        if u * (phi_max - phi_min) < phi_y(y + w) - phi_min:

                            # one of the test jump times \kappa_j is accepted and we reject the skeleton
                            test_factor_3 = False
                            break

                        j += 1

                    if test_factor_1 and test_factor_2 and test_factor_3:

                        # the sample is accepted
                        rejected = False

                        if stopping_time < T:
                            current_t = t
                            for jump in exponential_times:

                                while cap_counter < len(cap_limits) - 1:
                                    t_cap = cap_limits[cap_counter] - t
                                    if current_t < t + t_cap < t + jump:
                                        w = candidate_bridge[t_cap]
                                        cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                        print('cap added @ t=%s' % (t_cap + t))
                                        cap_counter += 1
                                    else:
                                        break

                                w = candidate_bridge[jump]
                                g = g * (1 - (f_inv(y + w) - x_down)/(x_up - x_down))
                                current_t += jump

                            while cap_counter < len(cap_limits) - 1:
                                if cap_limits[cap_counter] < stopping_time:
                                    t_cap = cap_limits[cap_counter] - t
                                    w = candidate_bridge[t_cap]
                                    cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                    print('cap added @ t=%s' % (t + t_cap))
                                    cap_counter += 1
                                else:
                                    break

                            g = g * np.exp(-x_down * tau)

                            y += candidate_bridge_values[-1]
                            t = stopping_time

                        else:
                            current_t = t

                            for jump in exponential_times:

                                if T - t < jump:
                                    break

                                while cap_counter < len(cap_limits) - 1:
                                    t_cap = cap_limits[cap_counter] - t
                                    if current_t < t + t_cap < t + jump:
                                        w = candidate_bridge[t_cap]
                                        cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                        print('cap added @ t=%s' % (t_cap + t))
                                        cap_counter += 1
                                    else:
                                        break

                                w = candidate_bridge[jump]
                                g = g * (1 - (f_inv(y + w) - x_down)/(x_up - x_down))
                                current_t += jump

                            while cap_counter < len(cap_limits) - 1:
                                t_cap = cap_limits[cap_counter] - t
                                w = candidate_bridge[t_cap]

                                cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                print('cap added @ t=%s' % (t + t_cap))
                                cap_counter += 1

                            w = candidate_bridge[T-t]
                            y += w
                            g = g * np.exp(-x_down * (T - t))
                            t = T

                            cap_value += g * max(f_inv(y)-strike, 0)
                            print('cap added @ t=3')

                else:
                    # one of the jump times was accepted and we test the skeleton up to the jump time
                    stopping_time = t + candidate_times[i]

                    u, v = np.random.rand(2)

                    # first Bernoulli variable
                    # K = -minimize_scalar(lambda x: - integrated_drift(x),
                    #                      bounds=(y-theta, y+theta), method='bounded').fun

                    # previously computed location of global maximum
                    K_max = 1.382436647017447

                    if y - theta < K_max < y + theta:
                        K = 0.011016
                    else:
                        K = max(integrated_drift(y - theta), integrated_drift(y + theta))

                    w = candidate_bridge[candidate_times[i]]
                    test_factor_1 = (np.log(u) < integrated_drift(y + w) - K)

                    # second Bernoulli variable
                    S = max(np.exp(-phi_min * (T - t)), 1)
                    test_factor_2 = (v * S < np.exp(-phi_min * candidate_times[i]))

                    test_factor_3 = True
                    j = 0
                    while j < len(test_times):

                        # we only need to test up to default time
                        if test_times[j] > candidate_times[i]:
                            break

                        u = np.random.rand()
                        w = candidate_bridge[test_times[j]]

                        if u * (phi_max - phi_min) < phi_y(y + w) - phi_min:
                            test_factor_3 = False
                            break

                        j += 1

                    if test_factor_1 and test_factor_2 and test_factor_3:
                        # we accept the skeleton up to jump time
                        rejected = False

                        current_t = t
                        for jump in exponential_times:
                            if candidate_times[i] < jump:
                                break

                            while cap_counter < len(cap_limits) - 1:
                                t_cap = cap_limits[cap_counter] - t
                                if current_t < t + t_cap < t + jump:
                                    w = candidate_bridge[t_cap]
                                    cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                    print('cap added @ t=%s' % (t_cap + t))
                                    cap_counter += 1
                                else:
                                    break

                            w = candidate_bridge[jump]
                            g = g * (1 - (f_inv(y + w) - x_down) / (x_up - x_down))
                            current_t += jump

                        while cap_counter < len(cap_limits) - 1:
                            if cap_limits[cap_counter] < stopping_time:
                                t_cap = cap_limits[cap_counter] - t
                                w = candidate_bridge[t_cap]
                                cap_value += g * max(f_inv(y + w) - strike, 0) * np.exp(-x_down * t_cap)
                                print('cap added @ t=%s' % (t + t_cap))
                                cap_counter += 1
                            else:
                                break

                        g = g * np.exp(- x_down * candidate_times[i])

                        # compute Y immediately before the jump
                        w = candidate_bridge[candidate_times[i]]
                        y += w

                        # compute Y after the jump
                        u = np.random.rand()
                        z = min_jump + (max_jump - min_jump) * u
                        y += delta_y(y, z)
                        t = stopping_time

        cap_results.append(cap_value)
        sim_count += 1
        if sim_count in sim_frames:
            print(sim_count)
            cap_price = np.mean(cap_results)
            sample_std = np.std(cap_results)
            std_error = sample_std / np.sqrt(sim_count)
            time_spent = time.time() - start_time
            sim_data.append([sim_count, cap_price, std_error, time_spent])
    return sim_data


# import cProfile
# cProfile.run('monte_carlo(n_sim=1000, sim_frames=[200,500,1000])')
# print(monte_carlo(n_sim=100000, sim_frames=[1000, 10000, 20000, 50000, 100000]))
