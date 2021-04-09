import numpy as np
from matplotlib import colors
from matplotlib.cm import get_cmap


def outline(image: np.ndarray, w: int = 1, c: str = "k", cmap_name: str = None, cmap_idx: int = 0) -> np.ndarray:
    """Draw an outline for image. The image has range (0,1).

    Besides color, one can use cmap to draw the color.

    Args:
        image (np.ndarray): input image
        w (int, optional): width. Defaults to 1.
        c (str, optional): color. Defaults to "k".
        cmap_name (str, optional): name of cmap. Defaults to None.
        cmap_idx (int, optional): color index. Defaults to 0.

    Returns:
        np.ndarray: outlined image

    """
    if cmap_name is not None:
        cmap_colors = get_cmap(cmap_name).colors
        color = cmap_colors[cmap_idx % len(cmap_colors)]
    else:
        color = colors.to_rgb(c)
    height, width, depth = image.shape
    new_image = np.zeros((height + 2 * w, width + 2 * w, depth))
    new_image[:w, :, :] = color
    new_image[-w:, :, :] = color
    new_image[:, :w, :] = color
    new_image[:, -w:, :] = color
    new_image[w:-w, w:-w, :] = image[:]
    return new_image
