import numpy as np
import sampling
from scipy.optimize import minimize_scalar
import time
import matplotlib.pyplot as plt


def monte_carlo(n_sim=1000, sim_frames=[1000], T=1, strike=5, X_0=50, beta=-1, r=0.05, sigma=50/4, b=0, c=1/2):

    # model functions
    # Lamperti transform
    def f(x):
        return (X_0**(-beta) - x**(-beta))/(beta*sigma)

    # inverse Lamperti transform
    def f_inv(x):
        return (X_0**(-beta) - x*sigma*beta)**(-1/beta)

    # drift function
    def mu_y(x):
        return ((r+b)/sigma)*(X_0**(-beta) - x*sigma*beta) + sigma * (c - (beta+1)/2)/(X_0**(-beta) - x*sigma*beta)

    def jump_int(x):
        return b + c * sigma**2 * x**(2*beta)

    def delta_y(x):
        return X_0**(-beta)/(sigma*beta) - x

    def phi_y(x):
        return 0.5 * (-(r+b)*beta + sigma**2 * beta * (c-(beta+1)/2)/(X_0**(-beta)-x*sigma*beta)**2 + mu_y(x)**2)

    # function A(y) needed for the acceptance tests
    def integrated_drift(x):
        return (r+b)/sigma * (X_0**(-beta)*x - x**2/2 * sigma * beta) - \
               (c - (beta+1)/2)/beta * np.log(1 - x*sigma*beta*X_0**beta)

    final_samples = []
    default_times = []
    sim_data = []
    optimal_thetas = [float(theta.strip()) for theta in open("jdcev_thetas.txt", 'r')]
    start_time = time.time()

    sim_count = 0
    while sim_count < n_sim:
        # print(sim_count)
        y = 0
        t = 0

        # main while loop iterating over time segments
        while t < T:
            # choice of theta
            y_range = np.linspace(-3.5, 20, 100)
            if -3.5 < y < 19:
                for i in range(len(y_range)):
                    if y_range[i] < y:
                        theta_opt = (optimal_thetas[i] * (y_range[i + 1] - y) + optimal_thetas[i + 1] * (
                                    y - y_range[i])) \
                                    / (y_range[i + 1] - y_range[i])
                        theta = min(theta_opt, (y - X_0 ** (-beta) / (sigma * beta)) / 2)
                        break
            else:
                theta = min(1.5, (y - X_0 ** (-beta) / (sigma * beta)) / 2)

            # sampling and acceptance while loop
            rejected = True
            while rejected:

                # determination of the minimum and maximum of phi
                # phi_max = -minimize_scalar(lambda x: -phi_y(x), bounds=(y-theta, y+theta), method='bounded').fun
                # phi_min = minimize_scalar(phi_y, bounds=(y-theta, y+theta), method='bounded').fun
                phi_max = phi_y(y+theta)
                # print('phi max = %s' % phi_max)
                phi_min = phi_y(y-theta)
                # print('phi min = %s' % phi_min)
                lamb = jump_int(f_inv(y - theta))

                # generate the exit time for the interval [y-theta, y+theta]
                tau = theta**2 * sampling.generate_exit_time()

                # generate the required Poisson jump times
                candidate_times = sampling.generate_poisson_jumps(lamb, min(tau, T-t))
                test_times = sampling.generate_poisson_jumps(phi_max-phi_min, min(tau, T-t))

                all_times = sorted(candidate_times + test_times)
                if T-t < tau:
                    all_times.append(T-t)
                all_times.append(tau)
                # generate a Brownian meander with exit time tau at all the required times and we rescale
                candidate_bridge_values = sampling.generate_brownian_meander([i/theta**2 for i in all_times])
                candidate_bridge_values = [x*theta for x in candidate_bridge_values]

                candidate_bridge = {all_times[i]: candidate_bridge_values[i] for i in range(len(all_times))}

                i = 0
                a = len(candidate_times)
                while i < a:
                    u = np.random.rand()
                    if u*lamb < jump_int(f_inv(y + candidate_bridge[candidate_times[i]])):
                        # a default time is accepted and we break the loop
                        break
                    else:
                        i += 1

                if i == a:

                    # no jump time is accepted and we test the complete skeleton
                    stopping_time = t + min(tau, T - t)

                    u, v = np.random.rand(2)

                    # first Bernoulli variable
                    K = integrated_drift(y+theta)
                    test_factor_1 = (np.log(u) < integrated_drift(y + candidate_bridge[min(tau, T-t)]) - K)
                    # second Bernoulli variable
                    S = max(np.exp(-phi_min*(T-t)), 1)
                    test_factor_2 = (v*S < np.exp(-phi_min*min(tau, T-t)))

                    # thinning test process
                    test_factor_3 = True
                    j = 0
                    while j < len(test_times):

                        w = np.random.rand()
                        if w * (phi_max-phi_min) < phi_y(y + candidate_bridge[test_times[j]])-phi_min:

                            # one of the test jump times \kappa_j is accepted and we reject the skeleton
                            test_factor_3 = False
                            break
                        j += 1

                    if test_factor_1 and test_factor_2 and test_factor_3:

                        # the sample is accepted
                        rejected = False
                        if stopping_time < T:
                            y += candidate_bridge_values[-1]
                            t = stopping_time

                        else:
                            y += candidate_bridge[T-t]
                            final_samples.append(f_inv(y))
                            t = T

                else:

                    # one of the jump times was accepted and we test the skeleton up to the default time
                    stopping_time = t + candidate_times[i]

                    u, v = np.random.rand(2)
                    K = integrated_drift(y+theta)
                    S = max(np.exp(-phi_min * min(tau, (T-t))), 1)
                    # first Bernoulli variable
                    test_factor_1 = (np.log(u) < integrated_drift(y + candidate_bridge[candidate_times[i]]) - K)
                    # second Bernoulli variable
                    test_factor_2 = (v*S < np.exp(-phi_min*candidate_times[i]))

                    test_factor_3 = True
                    j = 0
                    while j < len(test_times):

                        # we only need to test up to default time
                        if test_times[j] > candidate_times[i]:
                            break

                        w = np.random.rand()
                        if w * (phi_max-phi_min) < phi_y(y + candidate_bridge[test_times[j]])-phi_min:
                            test_factor_3 = False
                            break

                        j += 1

                    if test_factor_1 and test_factor_2 and test_factor_3:

                        rejected = False
                        final_samples.append(0)
                        default_times.append(stopping_time)
                        t = T
        sim_count += 1
        if sim_count in sim_frames:
            option_results = [max(strike - x, 0) for x in final_samples] * np.exp(-r)
            option_price = np.mean(option_results)
            sample_std = np.std(option_results)
            std_error = sample_std / np.sqrt(sim_count)
            time_spent = time.time() - start_time

            print(sim_count)
            sim_data.append([sim_count, option_price, std_error, time_spent])

    return sim_data


# import cProfile
# cProfile.run('monte_carlo(sim_frames=[200,500,10000],X_0=25)')
# print(monte_carlo(n_sim=10000, sim_frames=[200, 500, 1000, 10000], b=0.2))
