import numpy


def kl_bernoulli(mu_1, mu_2):
    return mu_1 * numpy.log(mu_1/mu_2) + (1-mu_1)*numpy.log((1-mu_1)/(1-mu_2))


# https://github.com/zakaryaxali/bandits_kl-ucb/blob/master/kl_ucb_policy.py
def klucb_upper_bisection(N, S, k, t, precision = 1e-6, max_iterations = 50):
    upperbound = numpy.log(t)/N[k]
    reward = S[k]/N[k]

    u = upperbound
    l = reward
    n = 0

    while n < max_iterations and u - l > precision:
        q = (l + u)/2
        if kl_bernoulli(reward, q) > upperbound:
            u = q
        else:
            l = q
        n += 1

    return (l+u)/2
