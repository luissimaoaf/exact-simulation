import numpy as np
import matplotlib.pyplot as plt
import sampling
import time


# model parameters

lamb0 = 0.0110
lamb1 = 0.1000

min_jump = 0.0113
max_jump = 0.0312

X_0 = 0.0422
mean = 0.0422
kappa = 0.0117
sigma = 0.0130


def jump_int(x):
    return lamb0 + lamb1 * max(x, 0)


T = 3
strike = 0.05
cap_limits = [1, 2, 3]


def monte_carlo(sim_frames=[1000]):
    sim_data = []
    for n_sim in sim_frames:

        sim_count = 0
        results = []

        N = int(np.sqrt(n_sim))
        h = T/N

        start_time = time.time()

        while sim_count < n_sim:
            i = 0
            g = 0
            cap_counter = 0
            bond_result = 0
            compensator = 0
            tau = sampling.generate_exp()

            X = X_0

            while i < N:
                u = np.random.normal()
                X_next = ((1 - kappa*h/2)*np.sqrt(X) + sigma*np.sqrt(h)*u/(2*(1 - kappa*h/2)))**2 + h*(kappa*mean - sigma**2/4)

                compensator += h * jump_int(X_next)
                if compensator > tau:
                    u = np.random.rand()
                    z = min_jump + (max_jump - min_jump)*u
                    X_next += z

                    tau += sampling.generate_exp()

                if i+1 >= N*cap_limits[cap_counter]/3:
                    bond_result += np.exp(-g)*max(X_next-strike, 0)
                    cap_counter += 1

                g += h * (X + X_next) / 2
                X = X_next
                i += 1

            results.append(bond_result)
            sim_count += 1

        print(sim_count)
        bond_price = np.mean(results)
        sample_std = np.std(results)
        std_error = sample_std/np.sqrt(n_sim)
        time_spent = time.time() - start_time

        sim_data.append([n_sim, bond_price, std_error, time_spent])

    return sim_data

