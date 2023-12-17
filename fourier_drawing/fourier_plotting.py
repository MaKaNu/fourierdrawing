import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


class VectorPlotter:
    def __init__(
        self,
        ax_limits=None,
        colors=None,
        end_color="r",
        end_markersize=3,
        connecting_style="-",
    ):
        self.ax_limits = ax_limits
        self.colors = colors
        self.end_color = end_color
        self.end_markersize = end_markersize
        self.endpoints = []
        self.connecting_style = connecting_style
        self.fig, self.ax = plt.subplots()

    def plot_vectors(self, vectors):
        current_endpoint = (0, 0)

        for length, angle in vectors:
            theta = np.radians(angle)

            endpoint = (
                current_endpoint[0] + length * np.cos(theta),
                current_endpoint[1] + length * np.sin(theta),
            )

            self.ax.quiver(
                *current_endpoint,
                endpoint[0] - current_endpoint[0],
                endpoint[1] - current_endpoint[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="b"
            )

            circle = plt.Circle(
                current_endpoint,
                length,
                color="b",
                fill=False,
                linestyle="dotted",
                linewidth=0.5,
            )
            self.ax.add_patch(circle)

            current_endpoint = endpoint

        self.endpoints.append(endpoint)

    def draw_all_endpoints(self):
        endpoints_np = np.array(self.endpoints)
        self.ax.scatter(
            endpoints_np[:, 0],
            endpoints_np[:, 1],
            self.end_markersize,
            color=self.end_color,
        )

    def draw_axis(self):
        endpoints_np = np.array(self.endpoints)
        if self.ax_limits is None:
            self.ax_limits = [
                -1 * np.max([np.abs(endpoints_np[:, 0]), np.abs(endpoints_np[:, 1])]),
                np.max([np.abs(endpoints_np[:, 0]), np.abs(endpoints_np[:, 1])]),
            ]

        self.ax.set_xlim([self.ax_limits[0], self.ax_limits[1]])
        self.ax.set_ylim([self.ax_limits[0], self.ax_limits[1]])

        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("Vector Plot with Circles")

        self.ax.grid(True)
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.set_aspect("equal", adjustable="box")

    def draw_connected_dots(self):
        endpoints_np = np.array(self.endpoints)
        self.ax.plot(
            endpoints_np[:, 0],
            endpoints_np[:, 1],
            linestyle=self.connecting_style,
            color=self.end_color,
        )

    def update(self, t, vectors):
        self.ax.clear()

        vectors_rot = vectors.copy()
        vectors_rot[0, 1] = vectors_rot[0, 1] + 0 * t
        for i in range(1, len(vectors), 2):
            vectors_rot[i, 1] = vectors_rot[i, 1] + i * t
            vectors_rot[i + 1, 1] = vectors_rot[i + 1, 1] - 1 * i * t

        self.plot_vectors(vectors_rot)
        self.draw_connected_dots()
        self.draw_all_endpoints()
        self.draw_axis()
