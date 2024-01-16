import numpy as np
import sampling
import time


def monte_carlo(sim_frames=[1000], T=1, strike=65, X_0=50, beta=-1,
                r=0.05, sigma=50/4, b=0, c=1/2, down_barrier=5):

    def jump_int(x):
        return b + c * sigma**2 * x**(2*beta)

    sim_data = []

    for n_sim in sim_frames:

        results = []

        N = int(np.sqrt(n_sim))
        h = T/N

        start_time = time.time()
        sim_count = 0
        while sim_count < n_sim:
            i = 0
            X = np.zeros(N+1)
            X[0] = X_0

            compensator = h * jump_int(X[0])
            default_breakpoint = sampling.generate_exp()

            while i < N:
                u = np.random.normal()
                X[i+1] = X[i] + (r + jump_int(X[i])) * X[i] * h + sigma*(X[i]**(beta+1)) * np.sqrt(h) * u

                compensator += h * jump_int(X[i+1])

                if X[i+1] < down_barrier or compensator > default_breakpoint:
                    X[i+1] = 0
                    break

                i += 1

            results.append(max(X[-1]-strike, 0))
            sim_count += 1

        print(n_sim)
        option_price = np.mean(results)*np.exp(-r)
        print(len(results))
        sample_std = np.std(results)
        std_error = sample_std/np.sqrt(n_sim)
        time_spent = time.time() - start_time

        sim_data.append([n_sim, option_price, std_error, time_spent])

    return sim_data

