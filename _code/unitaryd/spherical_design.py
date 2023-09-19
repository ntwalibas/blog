import numpy as np
import numpy.linalg as la

def tetrahedron():
    """The tetrahedron as a spherical 2-design."""
    polyhedron = np.array([
        [ 1,  0, -1/np.sqrt(2)],
        [-1,  0, -1/np.sqrt(2)],
        [ 0,  1,  1/np.sqrt(2)],
        [ 0, -1,  1/np.sqrt(2)],
    ])
    # Normalize so the tetrahedron fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in polyhedron]
    )

def cube():
    """The tetrahedron as the representative spherical 3-design."""
    polyhedron = np.array([
        [ 1,  1,  1], [-1,  1,  1],
        [-1, -1,  1], [-1, -1, -1],
        [ 1,  1, -1], [ 1, -1, -1],
        [-1,  1, -1], [ 1, -1,  1],
    ])
    # Normalize so the cube fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in polyhedron]
    )

def icosahedron():
    """The icosahedron as the representative spherical 5-design."""
    g = (1 + np.sqrt(5)) / 2
    polyhedron = np.array([
        [ 0,  1,  g], [ 0, -1,  g],
        [ 0,  1, -g], [ 0, -1, -g],

        [ 1,  g,  0], [-1,  g,  0],
        [ 1, -g,  0], [-1, -g,  0],

        [ g,  0,  1], [ g,  0, -1],
        [-g,  0,  1], [-g,  0, -1],
    ])
    # Normalize so the icosahedron fits into the unit sphere
    return np.array(
        [point / la.norm(point) for point in polyhedron]
    )

def spherical_design_average(f, points):
    """Computes the average of a function `f` using a spherical
    t-designs, specifically polyhedra.
    """
    return np.mean([f(*point) for point in points])

if __name__ == "__main__":
    f1 = lambda x, y, z: x**2 + x*z + y*z + z**2
    print("We expect all designs to provide the average:")
    print(f"f1 average using 2-design: {spherical_design_average(f1, tetrahedron())}")
    print(f"f1 average using 3-design: {spherical_design_average(f1, cube())}")
    print(f"f1 average using 5-design: {spherical_design_average(f1, icosahedron())}")

    f2 = lambda x, y, z: x**4
    print("\nWe expect the average to work only for the icosahedron:")
    print(f"f2 average using 2-design: {spherical_design_average(f2, tetrahedron())}")
    print(f"f2 average using 3-design: {spherical_design_average(f2, cube())}")
    print(f"f2 average using 5-design: {spherical_design_average(f2, icosahedron())}")
