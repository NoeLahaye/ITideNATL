### routine for preparing plots
import matplotlib as mpl
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np

_plot_kwgs = {"figsize":(12,8),
                 "central_lon":-40, "extent":[-95, 0, 6.5, 55]
                }               
                 

def prep_one_plot(**kwargs):
    kwgs = _plot_kwgs.copy()
    kwgs.update(kwargs)
    map_proj = ccrs.Robinson(central_longitude=kwgs["central_lon"])

    fig = plt.figure(figsize=kwgs["figsize"])
    ax = plt.axes(projection=map_proj)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.set_extent(kwgs["extent"])

    return fig, ax

def get_discrete_cmap_norm(vmin=0, vmax=1, nlevs=10, cmap="Spectral", extend="neither"):
    """ discrete colormap and linear norm

    linear colorbar between vmin and vmax with "nlevs" levels using colormap "cmap".
    Depending of the value of "extend", the colormap will have nlevs-1 (extend="neither"),
    nlevs (extend="min" or "max") or nlevs+1 (extend="both") colors.

    internally uses colors.from_levels_and_colors and pyplot.get_cmap from matplotlib

    returns
    _______
    cmap, norm

    """
    cmap = plt.get_cmap(cmap)

    if extend=="both":
        idp = 1
    elif extend=="neither":
        idp = -1
    else:
        idp = 0

    colors = cmap(np.linspace(0, 1, nlevs+idp))
    levels = np.linspace(vmin, vmax, nlevs)

    cmap, norm = mpl.colors.from_levels_and_colors(levels, colors, extend=extend)
    return cmap, norm
