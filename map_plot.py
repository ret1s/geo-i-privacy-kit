import contextily as ctx

import matplotlib.pyplot as plt
from v2 import get_road_network

def plot_road_network(G=None, gdf_tuple=None, location=None, distance=1000, figsize=(12, 12), add_basemap=True, title=None):
    # Get network data if not provided
    if (G is None or gdf_tuple is None) and location is not None:
        G, gdf_tuple = get_road_network(location, distance=distance)

    nodes_gdf, edges_gdf = gdf_tuple

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot edges (roads) and nodes (intersections)
    edges_gdf.plot(ax=ax, linewidth=1, color='black')
    nodes_gdf.plot(ax=ax, color='blue', alpha=0.7, markersize=5)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=15)

    # Remove axis borders
    ax.axis('off')

    # Add basemap if requested
    if add_basemap:
        try:
            ctx.add_basemap(ax, crs=edges_gdf.crs.to_string())
        except Exception as e:
            print(f"Failed to add basemap: {e}")

    # Set appropriate view limits
    margin = 0.01
    bounds = edges_gdf.total_bounds
    ax.set_xlim([bounds[0] - margin, bounds[2] + margin])
    ax.set_ylim([bounds[1] - margin, bounds[3] + margin])

    plt.tight_layout()
    return fig, ax