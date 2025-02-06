import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyzonotope.zonotope import Zonotope
from scipy.spatial import ConvexHull


class ZonotopePlotter3D:
    """
    A class for visualizing Zonotopes in 3D.
    """

    def __init__(self, zonotope: Zonotope, color='blue', alpha=0.5, projection_offset=0.2):
        self.zonotope = zonotope
        self.color = color
        self.alpha = alpha
        self.projection_offset = projection_offset

    def plot(self, ax=None):
        """ Plots the zonotope in 3D with optimized zoom and axis limits. """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        vertices = self.zonotope.compute_vertices()
        hull = self._compute_convex_hull(vertices)

        for simplex in hull:
            poly = Poly3DCollection([vertices[simplex]], alpha=self.alpha, facecolor=self.color, edgecolor='k')
            ax.add_collection3d(poly)

        # Compute axis limits
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)

        ax.set_xlim([min_vals[0], max_vals[0]])
        ax.set_ylim([min_vals[1], max_vals[1]])
        ax.set_zlim([min_vals[2], max_vals[2]])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        self._plot_projections(ax, vertices)

        return ax

    def _compute_convex_hull(self, vertices):
        """ Computes the convex hull faces for plotting. """
        hull = ConvexHull(vertices)
        return hull.simplices

    def _plot_projections(self, ax, vertices):
        """ Plots the convex hull of the projected points onto the XY, XZ, and YZ planes. """
        offset = self.projection_offset

        xy_proj = np.copy(vertices)
        xy_proj[:, 2] = min(vertices[:, 2]) - offset
        xy_hull = ConvexHull(xy_proj[:, :2])
        ax.add_collection3d(Poly3DCollection(xy_proj[xy_hull.simplices], alpha=0.6, facecolor='gray', edgecolor='k'))

        xz_proj = np.copy(vertices)
        xz_proj[:, 1] = min(vertices[:, 1]) - offset
        xz_hull = ConvexHull(xz_proj[:, [0, 2]])
        ax.add_collection3d(Poly3DCollection(xz_proj[xz_hull.simplices], alpha=0.6, facecolor='gray', edgecolor='k'))

        yz_proj = np.copy(vertices)
        yz_proj[:, 0] = min(vertices[:, 0]) - offset
        yz_hull = ConvexHull(yz_proj[:, [1, 2]])
        ax.add_collection3d(Poly3DCollection(yz_proj[yz_hull.simplices], alpha=0.6, facecolor='gray', edgecolor='k'))

# Example Usage
if __name__ == "__main__":
    center = np.array([1, 1, 1])
    generators = np.array([[-0.00012384,  0.000473   , -0.00018542],
                           [ 0.00012384,  0.00047301 ,  0.00018541],
                           [ 0.00012384,  0.         , -0.00037083]])
    
    zono = Zonotope(center, generators)
    plotter = ZonotopePlotter3D(zono, color='red', alpha=0.2, projection_offset=0.001)
    ax = plotter.plot()
    plt.show()
