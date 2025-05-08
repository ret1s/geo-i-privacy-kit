import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
import osmnx as ox
import networkx as nx
import random
from datetime import datetime

# Part 1: Geo-indistinguishability Mechanism
def planar_laplace_noise(epsilon):
    """Generate planar Laplace noise for geo-indistinguishability with parameter epsilon"""
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.exponential(1 / epsilon)
    return r * np.cos(theta), r * np.sin(theta)

# Part 2: Road Network Functions
def get_road_network(location, distance=3000):
    """Download road network for a specific location with given radius"""
    G = ox.graph_from_point(location, dist=distance, network_type='drive', simplify=False)
    
    # Add edge lengths if they don't exist
    if 'length' not in G.edges[list(G.edges)[0]]:
        try:
            G = ox.add_edge_lengths(G)
        except:
            for u, v, data in G.edges(data=True):
                if 'geometry' in data:
                    data['length'] = data['geometry'].length
                else:
                    # Calculate Euclidean distance
                    from_node = G.nodes[u]
                    to_node = G.nodes[v]
                    data['length'] = ((from_node['x'] - to_node['x'])**2 + 
                                     (from_node['y'] - to_node['y'])**2)**0.5
                    
    return G, ox.graph_to_gdfs(G, nodes=True, edges=True)

def get_nearest_node(G, point):
    """Get nearest node in graph to a point"""
    return ox.distance.nearest_nodes(G, point.x, point.y)

# Part 3: Helper Functions
def convert_nodes_to_points_and_edges(G, nodes_gdf, node_path):
    """Convert a node path to points and edges for visualization"""
    # Convert nodes to points
    trajectory_points = []
    for node_id in node_path:
        node = nodes_gdf.loc[node_id]
        trajectory_points.append(Point(node.geometry.x, node.geometry.y))
    
    # Create edges between consecutive points
    path_edges = []
    for i in range(len(node_path)-1):
        try:
            path = nx.shortest_path(G, node_path[i], node_path[i+1], weight='length')
            for u, v in zip(path[:-1], path[1:]):
                u_point = Point(nodes_gdf.loc[u].geometry.x, nodes_gdf.loc[u].geometry.y)
                v_point = Point(nodes_gdf.loc[v].geometry.x, nodes_gdf.loc[v].geometry.y)
                path_edges.append((u_point, v_point))
        except nx.NetworkXNoPath:
            continue
    
    return trajectory_points, path_edges

def get_road_weight(G, u, v):
    """Create weights that favor smaller roads"""
    data = G.get_edge_data(u, v, 0)
    road_type = data.get('highway', 'unclassified')
    
    # Create a bias for smaller roads
    if road_type in ['residential', 'living_street', 'service']:
        return data['length'] * 0.7  # Make smaller roads "shorter" 
    elif road_type in ['tertiary', 'unclassified']:
        return data['length'] * 0.85
    else:
        return data['length']  # Keep main roads at normal weight

# Part 4: Trajectory Generation Functions
def generate_trajectory(G, start_node, length=10, max_dist=1000):
    """Generate a simple random trajectory as a sequence of nodes"""
    trajectory = [start_node]
    current_node = start_node
    
    for _ in range(length-1):
        # Get neighboring nodes
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
            
        # Choose a random neighbor that's not already in the trajectory if possible
        available = [n for n in neighbors if n not in trajectory]
        if not available and neighbors:
            available = neighbors
            
        if available:
            next_node = random.choice(available)
            trajectory.append(next_node)
            current_node = next_node
        else:
            break
            
    return trajectory

def generate_complex_trajectory(G, start_node, length=30, randomness=0.3, min_distance=300):
    """
    Generate a more complex and realistic trajectory with turns and varied movement
    
    Parameters:
        randomness: How often to take a random turn (0-1)
        min_distance: Minimum distance to move before considering a direction change
    """
    trajectory = [start_node]
    current_node = start_node
    
    # Find distant nodes to use as destinations
    far_nodes = []
    try:
        center = nx.center(G)[0]
        dist = nx.single_source_dijkstra_path_length(G, center, weight='length')
        eligible_nodes = [node for node, d in dist.items() if d > min_distance and node != start_node]
        if eligible_nodes:
            far_nodes = random.sample(eligible_nodes, min(5, len(eligible_nodes)))
    except:
        far_nodes = []
    
    # Fallback if we couldn't get distant nodes
    if not far_nodes:
        nodes = list(G.nodes())
        if len(nodes) > 5 and start_node in nodes:
            nodes.remove(start_node)
            far_nodes = random.sample(nodes, min(5, len(nodes)))
    
    # Set up destination tracking
    destination_idx = 0
    current_destination = far_nodes[destination_idx] if far_nodes else None
    prev_direction = None
    
    for i in range(length-1):
        # Get neighboring nodes
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
            
        # Get available neighbors
        available = [n for n in neighbors if n not in trajectory]
        if not available and neighbors:
            available = neighbors
            
        if available:
            # Target destination with some probability
            if current_destination and np.random.random() > randomness and nx.has_path(G, current_node, current_destination):
                try:
                    path_to_dest = nx.shortest_path(G, current_node, current_destination, weight='length')
                    
                    if len(path_to_dest) > 1:
                        next_node = path_to_dest[1]
                    else:
                        # Reached destination, pick a new one
                        destination_idx = (destination_idx + 1) % len(far_nodes)
                        current_destination = far_nodes[destination_idx] if far_nodes else None
                        next_node = random.choice(available)
                        
                    trajectory.append(next_node)
                    current_node = next_node
                    
                    # Check if we've reached destination
                    if current_node == current_destination:
                        destination_idx = (destination_idx + 1) % len(far_nodes)
                        current_destination = far_nodes[destination_idx] if far_nodes else None
                except nx.NetworkXNoPath:
                    next_node = random.choice(available)
                    trajectory.append(next_node)
                    current_node = next_node
            else:
                # Move randomly, but try to maintain direction
                if prev_direction and len(available) > 1:
                    # Use directional bias to create smoother paths
                    weights = []
                    for n in available:
                        try:
                            # Get coordinates for direction calculation
                            curr_coords = (G.nodes[current_node]['x'], G.nodes[current_node]['y'])
                            prev_coords = (G.nodes[prev_direction]['x'], G.nodes[prev_direction]['y'])
                            next_coords = (G.nodes[n]['x'], G.nodes[n]['y'])
                            
                            # Calculate vectors and dot product
                            prev_vector = (curr_coords[0] - prev_coords[0], curr_coords[1] - prev_coords[1])
                            next_vector = (next_coords[0] - curr_coords[0], next_coords[1] - curr_coords[1])
                            dot = (prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1])
                            weight = max(0.1, dot)
                        except:
                            weight = 1.0
                            
                        weights.append(weight)
                    
                    # Normalize weights and choose next node
                    if sum(weights) > 0:
                        weights = [w/sum(weights) for w in weights]
                        
                    try:
                        next_node = random.choices(available, weights=weights, k=1)[0]
                    except:
                        next_node = random.choice(available)
                else:
                    next_node = random.choice(available)
                    
                prev_direction = current_node
                trajectory.append(next_node)
                current_node = next_node
        else:
            break
            
    return trajectory

# Add this helper function for calculating node distance
def calculate_node_distance(G, nodes_gdf, node1, node2):
    """Calculate the distance between two nodes in meters"""
    if node1 not in nodes_gdf.index or node2 not in nodes_gdf.index:
        return float('inf')
        
    # Get node coordinates
    n1 = nodes_gdf.loc[node1]
    n2 = nodes_gdf.loc[node2]
    
    # Convert to Web Mercator for accurate distance calculation
    p1 = gpd.GeoDataFrame(geometry=[n1.geometry], crs=nodes_gdf.crs).to_crs(epsg=3857)
    p2 = gpd.GeoDataFrame(geometry=[n2.geometry], crs=nodes_gdf.crs).to_crs(epsg=3857)
    
    # Calculate distance in meters
    return p1.distance(p2.iloc[0].geometry).values[0]

# Modify the generate_biased_trajectory function
def generate_biased_trajectory(G, nodes_gdf, real_trajectory_nodes, start_node, similarity_factor=0.7, length=10, end_focus=5, qos_radius=150):
    """
    Generate a trajectory that's biased toward the real trajectory,
    ensuring all points are within the QoS radius (150m) of corresponding real points
    
    Parameters:
        similarity_factor: 0.0 = completely random, 1.0 = follows real trajectory exactly
        end_focus: Number of final steps to bias toward the real end point vicinity
        qos_radius: Maximum allowed distance (m) between real and fake trajectory points
    """
    trajectory = [start_node]
    current_node = start_node
    
    # Get real end node
    real_end_node = real_trajectory_nodes[-1] if real_trajectory_nodes else None
    
    # Define distance constraints for end points
    min_final_distance = 2  # Minimum distance in edges
    max_final_distance = 3  # Maximum distance in edges
    
    # Create a point for deliberate path diversion
    diversion_point = random.randint(2, length // 2)
    
    # Generate trajectory
    for i in range(length-1):
        # Get corresponding real point (or use last available if beyond real length)
        real_idx = min(i+1, len(real_trajectory_nodes)-1)
        corresponding_real_node = real_trajectory_nodes[real_idx]
        
        # Adjust similarity based on position in trajectory
        current_similarity = similarity_factor
        steps_remaining = length - i - 1
        
        # First half - reduce similarity for diversity
        if i < length // 2:
            current_similarity = max(0.1, similarity_factor - 0.4)
            
            # Force random choice at diversion point
            if i == diversion_point:
                current_similarity = 0.0
        
        # Middle part - gradually restore similarity    
        elif steps_remaining > 3 and steps_remaining <= end_focus:
            current_similarity = similarity_factor * (1 - (steps_remaining / end_focus))
        
        # If within real trajectory length
        if i < len(real_trajectory_nodes) - 1:
            target_node = real_trajectory_nodes[i+1]
            
            # Near the end, target end point
            if 3 < steps_remaining <= end_focus and real_end_node:
                target_node = real_end_node
                
            # Get neighboring nodes
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
                
            # Filter neighbors that are within QoS radius from corresponding real point
            qos_valid_neighbors = []
            for n in neighbors:
                if n not in trajectory:  # Try to avoid revisiting nodes
                    distance = calculate_node_distance(G, nodes_gdf, n, corresponding_real_node)
                    if distance <= qos_radius:
                        qos_valid_neighbors.append(n)
            
            # If no valid neighbors within QoS radius, try allowing revisits
            if not qos_valid_neighbors:
                for n in neighbors:
                    distance = calculate_node_distance(G, nodes_gdf, n, corresponding_real_node)
                    if distance <= qos_radius:
                        qos_valid_neighbors.append(n)
            
            # If still no valid neighbors, try taking the closest available neighbor
            if not qos_valid_neighbors and neighbors:
                distances = [calculate_node_distance(G, nodes_gdf, n, corresponding_real_node) for n in neighbors]
                min_idx = distances.index(min(distances))
                qos_valid_neighbors = [neighbors[min_idx]]
            
            # Use the QoS-constrained neighbors as available options
            available = qos_valid_neighbors if qos_valid_neighbors else neighbors
            
            if available:
                # Deliberately divert from real path in first half
                if np.random.random() < 0.3 and i < length // 2:
                    if len(real_trajectory_nodes) > i+1:
                        real_next = real_trajectory_nodes[i+1]
                        avoid_nodes = []
                        
                        for n in available:
                            try:
                                # Find nodes that don't lead to real path but still within QoS radius
                                if nx.has_path(G, n, real_next):
                                    continue
                                avoid_nodes.append(n)
                            except:
                                avoid_nodes.append(n)
                                
                        if avoid_nodes:
                            next_node = random.choice(avoid_nodes)
                            trajectory.append(next_node)
                            current_node = next_node
                            continue
                
                # Normal biased choice
                if np.random.random() < current_similarity and nx.has_path(G, current_node, target_node):
                    try:
                        path_to_target = nx.shortest_path(G, current_node, target_node, weight='length')
                        
                        if len(path_to_target) > 1:
                            # Check if next node in path to target is within QoS radius
                            next_node_candidate = path_to_target[1]
                            distance = calculate_node_distance(G, nodes_gdf, next_node_candidate, corresponding_real_node)
                            
                            if distance <= qos_radius:
                                next_node = next_node_candidate
                            else:
                                # Choose a random node within QoS radius
                                next_node = random.choice(available)
                        else:
                            next_node = random.choice(available)
                            
                        trajectory.append(next_node)
                        current_node = next_node
                    except nx.NetworkXNoPath:
                        next_node = random.choice(available)
                        trajectory.append(next_node)
                        current_node = next_node
                else:
                    next_node = random.choice(available)
                    trajectory.append(next_node)
                    current_node = next_node
            else:
                break
        else:
            # Past real trajectory length
            # Use the last real node as reference for QoS constraint
            last_real_node = real_trajectory_nodes[-1]
            
            # Get neighboring nodes
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
                
            # Filter neighbors that are within QoS radius from last real point
            qos_valid_neighbors = []
            for n in neighbors:
                if n not in trajectory:  # Try to avoid revisiting nodes
                    distance = calculate_node_distance(G, nodes_gdf, n, last_real_node)
                    if distance <= qos_radius:
                        qos_valid_neighbors.append(n)
            
            # If no valid neighbors, use any neighbor within QoS radius
            if not qos_valid_neighbors:
                for n in neighbors:
                    distance = calculate_node_distance(G, nodes_gdf, n, last_real_node)
                    if distance <= qos_radius:
                        qos_valid_neighbors.append(n)
            
            # Use the QoS-constrained neighbors as available options
            available = qos_valid_neighbors if qos_valid_neighbors else neighbors
            
            if available:
                next_node = random.choice(available)
                trajectory.append(next_node)
                current_node = next_node
            else:
                break
    
    # Ensure appropriate distance from real end point
    # (while still maintaining QoS constraint)
    if real_end_node and trajectory:
        current_node = trajectory[-1]
        
        try:
            path_to_end = nx.shortest_path(G, current_node, real_end_node, weight='length')
            path_length = len(path_to_end) - 1
            
            # If too close to real end
            if path_length < min_final_distance:
                neighbors = list(G.neighbors(current_node))
                neighbors = [n for n in neighbors if n != real_end_node]
                
                if neighbors:
                    better_nodes = []
                    for n in neighbors:
                        try:
                            dist_to_end = len(nx.shortest_path(G, n, real_end_node)) - 1
                            qos_dist = calculate_node_distance(G, nodes_gdf, n, real_end_node)
                            
                            if min_final_distance <= dist_to_end <= max_final_distance and qos_dist <= qos_radius:
                                better_nodes.append(n)
                        except nx.NetworkXNoPath:
                            pass
                    
                    if better_nodes:
                        trajectory[-1] = random.choice(better_nodes)
                    elif neighbors:
                        # Find closest neighbor within QoS
                        qos_neighbors = []
                        for n in neighbors:
                            if calculate_node_distance(G, nodes_gdf, n, real_end_node) <= qos_radius:
                                qos_neighbors.append(n)
                        
                        if qos_neighbors:
                            trajectory[-1] = random.choice(qos_neighbors)
            
            # If too far from real end
            elif path_length > max_final_distance:
                if len(path_to_end) > min_final_distance + 1:
                    target_idx = len(path_to_end) - min_final_distance - 1
                    if 0 < target_idx < len(path_to_end):
                        # Check if the new endpoint is within QoS radius
                        potential_node = path_to_end[target_idx]
                        if calculate_node_distance(G, nodes_gdf, potential_node, real_end_node) <= qos_radius:
                            trajectory[-1] = potential_node
                        
        except nx.NetworkXNoPath:
            pass
    
    return trajectory

def create_visualization(real_location, real_traj, fake_trajs):
    """Create a 2x2 grid visualization comparing real and fake trajectories"""
    # Set up figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    axs = axs.flatten()
    
    # Colors for fake trajectories
    colors = ['blue', 'green', 'orange', 'purple']
    
    # Plot each fake trajectory in its own subplot
    for i, fake_traj in enumerate(fake_trajs):
        print(f"Creating subplot {i+1}/4...")
        ax = axs[i]
        fake_traj_points, fake_traj_edges = fake_traj
        
        # Create GeoDataFrames
        real_traj_gdf = gpd.GeoDataFrame(geometry=real_traj[0], crs="EPSG:4326").to_crs(epsg=3857)
        real_edge_lines = [LineString([u, v]) for u, v in real_traj[1]]
        real_edges_gdf = gpd.GeoDataFrame(geometry=real_edge_lines, crs="EPSG:4326").to_crs(epsg=3857)
        
        fake_traj_gdf = gpd.GeoDataFrame(geometry=fake_traj_points, crs="EPSG:4326").to_crs(epsg=3857)
        fake_edge_lines = [LineString([u, v]) for u, v in fake_traj_edges]
        fake_edges_gdf = gpd.GeoDataFrame(geometry=fake_edge_lines, crs="EPSG:4326").to_crs(epsg=3857)
        
        # Plot real trajectory
        real_edges_gdf.plot(ax=ax, color='red', linewidth=3, alpha=0.8, label='Real')
        real_traj_gdf.iloc[[0]].plot(ax=ax, color='lime', markersize=60, alpha=1.0, marker='^', 
                                    edgecolor='black', linewidth=2, label='Real Start')
        real_traj_gdf.iloc[[-1]].plot(ax=ax, color='magenta', markersize=60, alpha=1.0, marker='s', 
                                     edgecolor='black', linewidth=2, label='Real End')
        
        # Plot fake trajectory
        color = colors[i]
        fake_edges_gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.6, label=f'Fake {i+1}')
        fake_traj_gdf.iloc[[0]].plot(ax=ax, color='cyan', markersize=50, alpha=1.0, marker='^', 
                                    edgecolor='black', linewidth=2, label=f'Fake Start {i+1}')
        fake_traj_gdf.iloc[[-1]].plot(ax=ax, color='yellow', markersize=50, alpha=1.0, marker='s', 
                                     edgecolor='black', linewidth=2, label=f'Fake End {i+1}')
        
        # Calculate and set bounds
        minx_real, miny_real, maxx_real, maxy_real = real_edges_gdf.total_bounds
        minx_fake, miny_fake, maxx_fake, maxy_fake = fake_edges_gdf.total_bounds
        
        total_bounds = [
            min(minx_real, minx_fake),
            min(miny_real, miny_fake),
            max(maxx_real, maxx_fake),
            max(maxy_real, maxy_fake)
        ]
        
        padding = 3000
        ax.set_xlim(total_bounds[0] - padding, total_bounds[2] + padding)
        ax.set_ylim(total_bounds[1] - padding, total_bounds[3] + padding)
        
        # Add basemap
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=15)
        except Exception:
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15)
            except Exception as e:
                print(f"Error loading basemap: {e}")
        
        # Add labels
        ax.set_title(f"Real vs. Fake Trajectory {i+1}", fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.axis('off')
    
    # Add title and save figure
    fig.suptitle(f"Road-Constrained Privacy-Preserving Trajectories (ε = {epsilon}, similarity = {similarity_factor:.1f})", 
                fontsize=20, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"result/{current_time}_four_trajectories_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("Generated combined figure with 4 trajectory comparisons")

def main():
    """Main function to generate and visualize trajectories"""
    # Location and parameters
    real_location = (10.772, 106.698)  # Ho Chi Minh City (lat, lon)
    
    # Parameters for trajectory generation
    global epsilon, similarity_factor
    epsilon = 0.4
    scale_factor = 12.0
    similarity_factor = 0.2
    
    num_points = 35
    randomness_factor = 0.4
    network_distance = 6000 

    print(f"Generating trajectory with {num_points} points and similarity factor {similarity_factor}")
    
    # Generate road network and real trajectory
    G, (nodes_gdf, edges_gdf) = get_road_network(real_location, distance=network_distance)
    real_point = Point(real_location[1], real_location[0])
    real_start_node = get_nearest_node(G, real_point)
    
    print("Generating complex real trajectory...")
    real_trajectory_nodes = generate_complex_trajectory(G, real_start_node, length=num_points, 
                                                      randomness=randomness_factor)
    
    real_traj_points, real_traj_edges = convert_nodes_to_points_and_edges(G, nodes_gdf, real_trajectory_nodes)
    real_traj = (real_traj_points, real_traj_edges)
    
    # Generate 4 fake trajectories
    print("Generating 4 fake trajectories...")
    fake_trajs = []
    for i in range(4):
        print(f"Generating fake trajectory {i+1}/4...")
        
        # Apply geo-indistinguishability to starting point
        dx, dy = planar_laplace_noise(epsilon)
        dx *= scale_factor
        dy *= scale_factor
        
        # Convert noise from meters to degrees
        noisy_lat = real_location[0] + (dy / 111320)
        noisy_lon = real_location[1] + (dx / (40075000 * np.cos(np.radians(real_location[0])) / 360))
        
        # Find nearest node and generate trajectory
        noisy_point = Point(noisy_lon, noisy_lat)
        start_node = get_nearest_node(G, noisy_point)
        
        node_path = generate_biased_trajectory(
            G, nodes_gdf, real_trajectory_nodes, start_node,
            similarity_factor=similarity_factor, 
            length=num_points,
            end_focus=3,
            qos_radius=200
        )
        
        # Convert to points and edges
        trajectory_points, path_edges = convert_nodes_to_points_and_edges(G, nodes_gdf, node_path)
        fake_trajs.append((trajectory_points, path_edges))
    
    # Create visualization
    create_visualization(real_location, real_traj, fake_trajs)

if __name__ == "__main__":
    for i in range(5):
        print(f"Run {i+1}")
        main()
        print(f"✅ Completed run {i+1}")
