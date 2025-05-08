import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
import osmnx as ox
import networkx as nx
import random
from datetime import datetime

# ================= PLOTTING FUNCTIONS =================
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
def plot_trajectory(G, gdf_tuple, trajectory, title="Trajectory on Road Network", figsize=(12, 12)):
    # Plot the road network
    fig, ax = plot_road_network(G=G, gdf_tuple=gdf_tuple, title=title, figsize=figsize)

    # Convert trajectory to GeoDataFrame for plotting
    if trajectory:
        traj_points = [Point(lon, lat) for lat, lon in trajectory]
        traj_line = LineString([(lon, lat) for lat, lon in trajectory])

        # Create GeoDataFrame for the line
        traj_gdf = gpd.GeoDataFrame(geometry=[traj_line], crs=gdf_tuple[0].crs)

        # Create GeoDataFrame for all points in the trajectory
        all_points_gdf = gpd.GeoDataFrame(geometry=traj_points, crs=gdf_tuple[0].crs)

        # Plot the trajectory line
        traj_gdf.plot(ax=ax, color='red', linewidth=3, zorder=3)

        # Plot all trajectory points
        all_points_gdf.plot(ax=ax, color='orange', markersize=25, zorder=4, marker='o', alpha=0.7)

        # Also highlight the start and end points with different colors
        start_gdf = gpd.GeoDataFrame(geometry=[traj_points[0]], crs=gdf_tuple[0].crs)
        end_gdf = gpd.GeoDataFrame(geometry=[traj_points[-1]], crs=gdf_tuple[0].crs)

        start_gdf.plot(ax=ax, color='green', markersize=100, zorder=5, marker='o')
        end_gdf.plot(ax=ax, color='purple', markersize=100, zorder=5, marker='x')


    return fig, ax

# ================= LOGICAL FUNCTIONS =================
def planar_laplace_noise(epsilon):
    """Generate planar Laplace noise for geo-indistinguishability with parameter epsilon"""
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.exponential(1 / epsilon)
    return r * np.cos(theta), r * np.sin(theta)

def get_road_network(location, distance=3000, network_type='all', simplify=False):
    """Get the road network around a given location using OSMnx."""
    G = ox.graph_from_point(
        location,
        dist=distance,
        network_type=network_type,
        simplify=simplify
    )

    if 'length' not in G.edges[list(G.edges)[0]]:
        try:
            G = ox.add_edge_lengths(G)
        except:
            for u, v, data in G.edges(data=True):
                if 'geometry' in data:
                    data['length'] = data['geometry'].length
                else:
                    from_node = G.nodes[u]
                    to_node = G.nodes[v]
                    data['length'] = ((from_node['x'] - to_node['x'])**2 + (from_node['y'] - to_node['y'])**2)**0.5

    return G, ox.graph_to_gdfs(G, nodes=True, edges=True)

def generate_trajectory(G, num_points=50, random_start=True, start_point=None, end_point=None, min_length=500):
    valid_nodes = [node for node, data in G.nodes(data=True) if 'x' in data and 'y' in data]

    if len(valid_nodes) < 2:
        raise ValueError("Not enough valid nodes in the graph")

    if random_start or start_point is None:
        source = random.choice(valid_nodes)
    else:
        source = start_point

    if end_point is None:
        path_length = 0
        attempts = 0

        while path_length < min_length and attempts < 20:
            target = random.choice(valid_nodes)

            if target == source:
                continue

            try:
                path = nx.shortest_path(G, source, target, weight='length')

                path_length = sum(G[u][v][0]['length'] for u, v in zip(path[:-1], path[1:]))

                if path_length >= min_length:
                    break
            except nx.NetworkXNoPath:
                pass

            attempts += 1
    else:
        target = end_point
        path = nx.shortest_path(G, source, target, weight='length')

    if 'path' not in locals():
        raise ValueError("Could not find a suitable path")

    path_edges = list(zip(path[:-1], path[1:]))

    points = []
    edge_points = []

    for u, v in path_edges:
        data = G[u][v][0]

        if 'geometry' in data:
            line = data['geometry']
            coords = list(line.coords)
            edge_points.extend(coords)
        else:
            start_x, start_y = G.nodes[u]['x'], G.nodes[u]['y']
            end_x, end_y = G.nodes[v]['x'], G.nodes[v]['y']
            edge_points.append((start_x, start_y))
            edge_points.append((end_x, end_y))

    unique_points = []
    for p in edge_points:
        if not unique_points or p != unique_points[-1]:
            unique_points.append(p)

    if len(unique_points) >= 2:
        total_dist = 0
        for i in range(len(unique_points) - 1):
            x1, y1 = unique_points[i]
            x2, y2 = unique_points[i+1]
            segment_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            total_dist += segment_dist

        points = []
        points_per_meter = (num_points - 1) / total_dist if total_dist > 0 else 1

        for i in range(len(unique_points) - 1):
            x1, y1 = unique_points[i]
            x2, y2 = unique_points[i+1]
            segment_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

            segment_points = max(1, int(segment_dist * points_per_meter))

            for j in range(segment_points):
                if i == 0 and j == 0:
                    t = 0
                else:
                    t = (j + 1) / segment_points

                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                points.append((y, x))

    if unique_points and (unique_points[-1][1], unique_points[-1][0]) not in points:
        x, y = unique_points[-1]
        points.append((y, x))

    return points, path_edges

# ================= TESTING FUNCTIONS =================
def example_trajectory():
    """Generate and plot an example trajectory in Saigon"""
    location = (40.7580, -73.9855)  # Times Square, New York City, USA

    # Get the road network with all road types
    G, gdfs = get_road_network(location, distance=2000, network_type='drive')

    # Generate a random trajectory
    trajectory, path_edges = generate_trajectory(G, num_points=100, min_length=1000)

    # Plot the trajectory on the road network
    fig, ax = plot_trajectory(G, gdfs, trajectory, title="Example Trajectory in New York City", figsize=(12, 12))

    plt.show()

    return trajectory

example_trajectory()