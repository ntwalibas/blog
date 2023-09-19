import numpy as np

def monte_carlo_average(f, sample_size):
    """Compute the average of a function `f` over the unit circle.
    """
    def sample_from_circle(size):
        x_samples = np.random.uniform(0, 2 * np.pi, size)
        y_samples = np.random.uniform(0, 2 * np.pi, size)
        return zip(
            np.cos(x_samples),
            np.sin(y_samples)
        )

    return np.mean([f(*sample) for sample in sample_from_circle(sample_size)])

if __name__ == "__main__":
    f = lambda x, y: x**2
    print(monte_carlo_average(f, 100_000))
