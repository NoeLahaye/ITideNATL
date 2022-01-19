### routine for preparing plots
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plot_kwgs = {"figsize":(12,8),
                 "central_lon":-40, "extent":[-95, 0, 6.5, 55]
                }               
                 
def prep_one_plot(**kwargs):
    kwgs = plot_kwgs.copy()
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
