import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import joblib
from python.utils import DataClean


class AnimatedQuiver(DataClean):
    """An animated scatter plot using matplotlib.animations.FuncAnimation.
        inspired by `https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot`
    """
    def __init__(self, coordinates:dict, fields:dict, field_label: str = 'U', criterion=1e-20):
        super().__init__(criterion)
        self.coordinates = coordinates
        self.fields = fields
        self.field_label = field_label
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.coordinates.keys()),
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        cc = next(iter(self.coordinates.values()))
        xy, uv = next(self.stream)
        self.artist = self.ax.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1])
        self.ax.set_xlim(cc[:,0].min(), cc[:,0].max())
        self.ax.set_ylim(cc[:,1].min(), cc[:,1].max())
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.artist,

    def data_stream(self):
        for time in self.coordinates.keys():
            coords = self.coordinates[time]
            field = self.fields[time][self.field_label]
            xy, field = self.filter_2d(coords, field)
            yield xy, field

    def update(self, i):
        """Update the scatter plot."""
        xy, uv = next(self.stream)

        # Set x and y data...
        self.artist.set_offsets(xy)
        # Set colors..
        self.artist.set_UVC(uv[:,0],uv[:,1])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.artist,


if __name__ == '__main__':
    co = joblib.load('data/thin_p_coordinates.pkl')
    fi = joblib.load('data/thin_p_fields.pkl')
    a = AnimatedQuiver(co, fi)
    plt.show()