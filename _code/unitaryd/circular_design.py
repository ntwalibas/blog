import numpy as np
import numpy.linalg as la

def circular_design_average(f, t):
    """Computes the average of a function `f` using circular
    t-designs, specifically polygons.
    """
    def polygon(number_of_sides):
        """
        Given the number of sides, this function returns
        the coordinates of the vertex corners of the polygon with
        the given number of sides.
        """
        internal_angle = 2 * np.pi / number_of_sides
        coordinates = []
        
        for point in range(number_of_sides):
            new_angle = point * internal_angle
            coordinates.append([np.cos(new_angle), np.sin(new_angle)])
        
        return coordinates

    return np.mean([f(*point) for point in polygon(t + 1)])

if __name__ == "__main__":
    f = lambda x, y: x**8 + (x**5)*(y**3) + x*(y**7)
    for vertex_count in range(2, 15):
        print(f"Design = {vertex_count + 1}\taverage = {circular_design_average(f, vertex_count)}")
