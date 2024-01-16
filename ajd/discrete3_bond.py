import numpy as np
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


def monte_carlo(sim_frames=[1000]):

    sim_data = []
    for n_sim in sim_frames:
        results = []
        N = int(n_sim**(1/4))
        h = T / N

        start_time = time.time()
        sim_count = 0
        while sim_count < n_sim:
            i = 0
            g = 0

            compensator = 0
            tau = sampling.generate_exp()

            X = X_0

            while i < N + 1:
                u = np.random.normal()
                X_next = np.exp(-kappa * h / 2) * (np.sqrt((mean * kappa - sigma ** 2 / 4) *
                                                           (1 - np.exp(-kappa * h / 2)) / kappa +
                                                           np.exp(-kappa * h / 2) * X) + sigma * np.sqrt(h) * u / 2) ** 2 \
                         + (kappa*mean - sigma**2/4)*(1 - np.exp(kappa*h/2))/kappa

                compensator += h * jump_int(X_next)
                if compensator > tau:
                    u = np.random.rand()
                    z = min_jump + (max_jump - min_jump) * u
                    X_next += z

                    tau += sampling.generate_exp()

                g += h * (X + X_next) / 2
                X = X_next
                i += 1

            bond_result = np.exp(-g)
            results.append(bond_result)
            sim_count += 1

        print(n_sim)
        bond_price = np.mean(results)
        sample_std = np.std(results)
        std_error = sample_std / np.sqrt(n_sim)
        time_spent = time.time() - start_time

        sim_data.append([n_sim, bond_price, std_error, time_spent])

    return sim_data
