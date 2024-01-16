import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import gamma


def generate_exp(lamb=1):
    # generates an exponential random variable with the inverse transform method
    u = np.random.rand()
    sample = - np.log(u)/lamb
    return sample


def generate_poisson_jumps(lamb, T):
    # generates poisson jump times (up to T) with rate lamb
    jump_times = []
    t = 0
    while t < T:
        tau = generate_exp(lamb)
        t += tau
        if t < T:
            jump_times.append(t)

    return jump_times


def generate_bm(times):
    # generates a finite sample path of Brownian motion at the requested times

    n = len(times)

    z = np.random.normal()
    new_step = np.sqrt(times[0]) * z
    path = [new_step]

    prev_step = new_step
    i = 1
    while i < n:
        z = np.random.normal()
        dt = times[i] - times[i-1]
        new_step = prev_step + np.sqrt(dt)*z

        path.append(new_step)
        prev_step = new_step

        i += 1

    return path


def generate_bridge(times):
    # generates a sample path of a Brownian bridge from a sample path of brownian motion

    n = len(times)
    T = times[-1]

    bm = generate_bm(times)
    bridge = []

    for i in range(n):
        new_step = bm[i] - times[i] * bm[-1] / T
        bridge.append(new_step)

    return bridge


def gamma_dens(t, b, y):
    # pdf of the gamma distribution

    g = y**b * t**(b-1) * np.exp(-y*t) / gamma(b)
    return g


def generate_exit_time():
    # generates a sample of Brownian motion exit time following the approach of

    a = 1.243707
    b = 1.088870
    y = 1.233701

    tau = 0
    rejected = True

    while rejected:
        v = np.random.gamma(shape=b, scale=1 / y)
        u = np.random.rand()
        test_value = a * u * gamma_dens(v, b, y)

        h = 1 / np.sqrt(2 * np.pi * v ** 3) * np.exp(-1 / (2 * v))
        terminated = False
        j = 1

        while not terminated:

            p_term = (2 * j + 1) * np.exp(-(2 * j + 1) ** 2 / (2 * v))
            n_term = (2 * j - 1) * np.exp(-(2 * j - 1) ** 2 / (2 * v))
            h_next = h + (-1) ** j / np.sqrt(2 * np.pi * v ** 3) * (p_term - n_term)

            if test_value < h_next <= h:
                terminated = True
                rejected = False

                tau = v

            elif h <= h_next < test_value:
                terminated = True

            h = h_next
            j += 1

    return tau


def generate_brownian_meander(times):
    # generates a sample path of a Brownian meander at the given times, where the last element is taken as the exit time
    # follows the approach of

    tau = times[-1]

    meander = []
    rejected = True  # set to True so that we go through the loop at least once
    invalid = False

    # if only the exit time is given, we return the final value (-1 or 1 with equal probability)
    if len(times) == 1:
        w_tau = np.random.choice([-1, 1])
        return [w_tau]

    while rejected or invalid:
        invalid = False
        rejected = False

        bridges = [[], [], []]
        candidate = []
        w_tau = np.random.choice([-1, 1])

        # generate the 3 required Brownian bridges
        for i in range(3):
            bridges[i] = generate_bridge(times)

        # transform them into a test sample
        for i in range(len(times)):
            t = times[i]
            b = np.sqrt(((tau - t) / tau + bridges[0][i]) ** 2 + bridges[1][i] ** 2 + bridges[2][i] ** 2)

            if b >= 2:
                # the sample is invalid, and we break out of the for loop
                invalid = True
                break

            candidate.append(b)

        # if our sample is invalid, we return to the beginning of the while loop
        if invalid:
            continue

        # first part of the test (acceptance/rejection against p)
        for i in range(len(times) - 2):
            u = np.random.rand()

            s = tau - times[i]
            x = candidate[i]
            t = tau - times[i + 1]
            y = candidate[i + 1]

            denom = 1 - np.exp(2 * x * y / (t - s))
            p = 1
            terminated = False
            j = 1

            # only a finite number of steps is needed for our verification to terminate
            while not terminated:

                if j % 2 == 0:
                    jj = j // 2
                    nu = np.exp(2 * jj * (4 * jj + 2 * (x - y)) / (t - s)) + np.exp(
                        2 * jj * (4 * jj - 2 * (x - y)) / (t - s))
                    p_next = p + nu
                else:
                    jj = j // 2 + 1
                    theta = np.exp(2 * (2 * jj - x) * (2 * jj - y) / (t - s)) + np.exp(
                        2 * (2 * (jj - 1) + x) * (2 * (jj - 1) + y) / (t - s))
                    p_next = p - theta

                if denom * u < p_next <= p:
                    # the sum terminates and we don't reject the sample
                    terminated = True

                elif p <= p_next < u:
                    # the sum terminates and we reject the sample
                    rejected = True
                    terminated = True

                p = p_next
                j += 1

            # if the sample was already rejected, we break out of the for loop
            if rejected:
                break

        # if we reject the sample in the first tests, we return to the beginning of the while loop
        if rejected:
            continue

        # second part of the test (acceptance/rejection against q)
        u = np.random.rand()
        t = tau - times[-2]
        x = candidate[-2]

        q = 1
        j = 1
        terminated = False

        while not terminated:

            if j % 2 == 0:
                jj = j // 2
                rho2 = (4 * jj + x) * np.exp(-4 * jj * (2 * jj + x) / t)
                q_next = q + rho2 / x
            else:
                jj = j // 2 + 1
                rho1 = (4 * jj - x) * np.exp(-4 * jj * (2 * jj - x) / t)
                q_next = q - rho1 / x

            if u < q_next <= q:
                # the sum terminates and we do accept the sample
                terminated = True
            elif q <= q_next < u:
                # the sum terminates and we reject the sample
                terminated = True
                rejected = True

            q = q_next
            j += 1

    if w_tau == 1:
        meander = [1 - b for b in candidate]
    else:
        meander = [b - 1 for b in candidate]

    return meander

# while __name__ == '__main__':
#
#     N = 10
#     count = 0
#     samples = []
#
#     while count < N:
#         meander = generate_brownian_meander(np.linspace(0, 2, 100))
#         plt.plot(meander)
#         samples.append(meander[-1])
#         count += 1
#
#
#     plt.show()
#     plt.hist(samples)
#     plt.show()
#     sys.exit()



