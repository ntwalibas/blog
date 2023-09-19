import numpy as np

def monte_carlo_average(f, a, b, sample_size):
    """Compute the average of a function `f` in the interval [a,b)
    The interval is right-open because of the implementation of
    `np.random.uniform` but it doesn't affect our final result
    because we just need to get arbitrarily close to  `b`.
    """
    if b - a == 0:
        raise ValueError(f"Cannot compute the average in the interval [{a},{b}].")

    samples = np.random.uniform(-1, 1, sample_size)
    total = 0
    for sample in samples:
        total += sample
    return total / len(samples)

if __name__ == "__main__":
    f = lambda x: 4 - x**2
    print(monte_carlo_average(f, -1, 1, 100_000))