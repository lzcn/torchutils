import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


# https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
def imscatter(x: float, y: float, image: np.ndarray, ax: Axes, zoom: float = 1) -> Artist:
    """Plot an image in give point

    Args:
        x (float): x
        y (float): y
        image (np.ndarray): image
        ax (Axes): axes to plot
        zoom (float, optional): scale for image. Defaults to 1.
    """
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    return artists
