import numpy as np
import osmnx as ox
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from geopy.distance import geodesic
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

def haversine_distance(p1, p2):
    return geodesic(p1, p2).meters

def bearing(p1, p2):
    lat1, lon1 = np.radians(p1)
    lat2, lon2 = np.radians(p2)
    d_lon = lon2 - lon1
    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(d_lon)
    angle = np.arctan2(x, y)
    return angle % (2 * np.pi)

def generate_candidates_fast(p, R, N):
    lat, lon = p
    u = np.random.uniform(0, 1, N)
    v = np.random.uniform(0, 1, N)
    r = R * np.sqrt(u)
    alpha = 2 * np.pi * v
    d_lat = (r / 6378137) * (180 / np.pi)
    d_lon = d_lat / np.cos(np.radians(lat))
    lat_cands = lat + d_lat * np.cos(alpha)
    lon_cands = lon + d_lon * np.sin(alpha)
    return np.column_stack((lat_cands, lon_cands))

def snap_to_road_fast(candidate, edges):
    point_geom = Point(candidate[1], candidate[0])
    nearest_edge = min(edges, key=lambda edge: point_geom.distance(edge))
    snapped_point = nearest_points(nearest_edge, point_geom)[0]
    return (snapped_point.y, snapped_point.x)

def prepare_road_edges(G):
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' in data:
            edges.append(data['geometry'])
        else:
            u_point = Point((G.nodes[u]['x'], G.nodes[u]['y']))
            v_point = Point((G.nodes[v]['x'], G.nodes[v]['y']))
            edges.append(LineString([u_point, v_point]))
    return edges

def AAGITP_fast(trajectory, R, epsilon, lambd, N, G):
    perturbed_traj = []
    edges = prepare_road_edges(G)  # chuẩn bị trước các cạnh đường
    for i, p in enumerate(trajectory):
        theta_real = bearing(trajectory[i-1], p) if i > 0 else None
        for attempt in range(5):  # Tối đa 5 lần thử
            candidates = generate_candidates_fast(p, R, N)
            distances = np.array([haversine_distance(p, cand) for cand in candidates])
            if theta_real is not None:
                angles = np.array([bearing(trajectory[i-1], cand) for cand in candidates])
                delta_angles = np.minimum(abs(angles - theta_real), 2*np.pi - abs(angles - theta_real))
            else:
                delta_angles = np.zeros(N)

            weights = np.exp(-epsilon * distances - lambd * delta_angles)
            probs = weights / weights.sum()
            chosen_index = np.random.choice(N, p=probs)
            candidate_chosen = candidates[chosen_index]

            candidate_snapped = snap_to_road_fast(candidate_chosen, edges)
            if haversine_distance(p, candidate_snapped) <= R:
                perturbed_traj.append(candidate_snapped)
                break
        else:
            perturbed_traj.append(p)  # Nếu thất bại, dùng điểm gốc
    return perturbed_traj

def plot_trajectories(real_traj, fake_traj):
    real_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in real_traj], crs="EPSG:4326")
    fake_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in fake_traj], crs="EPSG:4326")

    real_gdf = real_gdf.to_crs(epsg=3857)
    fake_gdf = fake_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Vẽ quỹ đạo thực tế
    real_gdf.plot(ax=ax, color='blue', markersize=60, alpha=0.8, label='Real points')
    ax.plot(real_gdf.geometry.x, real_gdf.geometry.y, color='blue', linewidth=2, linestyle='-', label='Real trajectory')

    # Vẽ quỹ đạo giả
    fake_gdf.plot(ax=ax, color='red', markersize=60, alpha=0.8, label='Perturbed points')
    ax.plot(fake_gdf.geometry.x, fake_gdf.geometry.y, color='red', linewidth=2, linestyle='--', label='Perturbed trajectory')

    # Thêm basemap từ contextily
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.legend()
    plt.title('Real vs Perturbed Trajectory')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()

# Ví dụ chạy thuật toán nhanh hơn:
if __name__ == "__main__":
    import random

    center_point = (np.random.uniform(40.477399, 40.917577), np.random.uniform(-74.25909, -73.700272))  # Random point in NYC
    G = ox.graph_from_point(center_point, dist=5000, network_type='drive')

    # Generate realistic initial points on road network
    def generate_realistic_trajectory(G, start_point, num_points=10, step_length=100):
        nodes, _ = ox.graph_to_gdfs(G)
        nearest_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])

        path = [nearest_node]
        current_node = nearest_node

        for _ in range(num_points - 1):
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            current_node = random.choice(neighbors)
            path.append(current_node)

        trajectory = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
        return trajectory

    # Generate trajectory on actual roads
    real_trajectory = generate_realistic_trajectory(G, center_point, num_points=10, step_length=100)

    R = 100
    epsilon = 0.01
    lambd = 1.0
    N = 20  # giảm số ứng viên để nhanh hơn rất nhiều

    perturbed_trajectory = AAGITP_fast(real_trajectory, R, epsilon, lambd, N, G)

    for orig, pert in zip(real_trajectory, perturbed_trajectory):
        print(f"Original: {orig} --> Perturbed: {pert}")

    plot_trajectories(real_trajectory, perturbed_trajectory)
