import numpy as np

def monte_carlo_average(f, sample_size):
    """Compute the average of a function `f` over the unit sphere
    """
    def sample_from_sphere(size):
        theta = np.random.uniform(0, 2 * np.pi, size)
        p = np.random.uniform(-1, 1, size)
        phi = np.arccos(p)
        return zip(
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.cos(theta),
            np.cos(phi)
        )

    return np.mean([f(*sample) for sample in sample_from_sphere(sample_size)])

if __name__ == "__main__":
    f = lambda x, y, z: x**4
    print(monte_carlo_average(f, 100_000))
