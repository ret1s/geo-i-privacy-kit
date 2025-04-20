import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
import osmnx as ox
import networkx as nx
import random
from datetime import datetime

# Part 1: Geo-indistinguishability Mechanism - Planar Laplace Noise
def planar_laplace_noise(epsilon):
    """Generate planar Laplace noise for geo-indistinguishability with parameter epsilon"""
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.exponential(1 / epsilon)
    return r * np.cos(theta), r * np.sin(theta)

# Part 2: Road Network Functions
def get_road_network(location, distance=3000):
    """Download road network for a specific location with given radius"""
    G = ox.graph_from_point(location, dist=distance, network_type='drive')
    
    # In newer versions of OSMnx, edge lengths are added automatically
    # Check if edge lengths exist, if not, add them
    if 'length' not in G.edges[list(G.edges)[0]]:
        try:
            # For newer versions
            G = ox.add_edge_lengths(G)
        except:
            # Alternative approach if the above fails
            for u, v, data in G.edges(data=True):
                if 'geometry' in data:
                    data['length'] = data['geometry'].length
                else:
                    # Calculate Euclidean distance if geometry not available
                    from_node = G.nodes[u]
                    to_node = G.nodes[v]
                    data['length'] = ((from_node['x'] - to_node['x'])**2 + 
                                      (from_node['y'] - to_node['y'])**2)**0.5
                    
    return G, ox.graph_to_gdfs(G, nodes=True, edges=True)

def get_nearest_node(G, point):
    """Get nearest node in graph to a point"""
    return ox.distance.nearest_nodes(G, point.x, point.y)

# Part 3: Trajectory Generation Functions
def generate_trajectory(G, start_node, length=10, max_dist=1000):
    """Generate a random trajectory as a sequence of nodes in the graph"""
    trajectory = [start_node]
    current_node = start_node
    
    for _ in range(length-1):
        # Get neighboring nodes
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
            
        # Choose a random neighbor that's not already in the trajectory (if possible)
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

def generate_real_trajectory(location, num_points=15):
    """Generate a real trajectory starting from exact location"""
    # Get road network
    print("Downloading road network...")
    G, (nodes_gdf, edges_gdf) = get_road_network(location, distance=5000)
    
    # Find nearest node to real location
    real_point = Point(location[1], location[0])
    start_node = get_nearest_node(G, real_point)
    
    # Generate trajectory
    print("Generating real trajectory...")
    node_path = generate_trajectory(G, start_node, length=num_points)
    
    # Convert node IDs to points
    trajectory_points = []
    for node_id in node_path:
        node = nodes_gdf.loc[node_id]
        trajectory_points.append(Point(node.geometry.x, node.geometry.y))
    
    # Create a complete path with edges
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

def generate_biased_trajectory(G, nodes_gdf, real_trajectory_nodes, start_node, similarity_factor=0.7, length=10, end_focus=5):
    """
    Generate a trajectory that's biased toward the real trajectory,
    but with more diversity in the middle portion
    
    Parameters:
        similarity_factor: 0.0 = completely random, 1.0 = follows real trajectory exactly
        end_focus: Number of final steps to bias toward the real end point vicinity
    """
    trajectory = [start_node]
    current_node = start_node
    
    # Get the real trajectory's end node for special handling
    real_end_node = real_trajectory_nodes[-1] if real_trajectory_nodes else None
    
    # Define minimum and maximum distance from real end point
    min_final_distance = 2  # Minimum distance in edges (to avoid ending at same point)
    max_final_distance = 3  # Maximum distance in edges (to stay nearby)
    
    # Determine a point in the trajectory to deliberately divert
    diversion_point = random.randint(2, length // 2)  # Divert somewhere in first half
    
    # Process for regular steps
    for i in range(length-1):
        # Determine current similarity - with strategic diversity
        current_similarity = similarity_factor
        steps_remaining = length - i - 1
        
        # Near the beginning and middle, reduce similarity for diversity
        if i < length // 2:
            # First half of trajectory - significantly lower similarity
            current_similarity = max(0.1, similarity_factor - 0.4)
            
            # At the diversion point, force a random turn
            if i == diversion_point:
                current_similarity = 0.0  # Force random choice
        
        # Middle part - gradually restore similarity    
        elif steps_remaining > 3 and steps_remaining <= end_focus:
            # Gradually increase similarity as we approach end focus area
            current_similarity = similarity_factor * (1 - (steps_remaining / end_focus))
        
        # Standard trajectory generation
        if i < len(real_trajectory_nodes) - 1:
            # Target next node in real trajectory
            target_node = real_trajectory_nodes[i+1]
            
            # Near the end (but not at final 3 steps), target vicinity of end
            if 3 < steps_remaining <= end_focus and real_end_node:
                target_node = real_end_node
                
            # Get neighboring nodes
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
                
            available = [n for n in neighbors if n not in trajectory]
            if not available and neighbors:
                available = neighbors
                
            if available:
                # Sometimes deliberately move away from the real path
                if np.random.random() < 0.3 and i < length // 2:  # 30% chance in first half
                    # Try to find nodes that move away from real path
                    if len(real_trajectory_nodes) > i+1:
                        real_next = real_trajectory_nodes[i+1]
                        avoid_nodes = []
                        
                        for n in available:
                            try:
                                # Prefer nodes that don't move toward the real path
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
                    # Try to move toward target
                    try:
                        path_to_target = nx.shortest_path(G, current_node, target_node, weight='length')
                        
                        if len(path_to_target) > 1:
                            next_node = path_to_target[1]
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
            # We've gone past the real trajectory length
            # Get available neighbors
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
                
            available = [n for n in neighbors if n not in trajectory]
            if not available and neighbors:
                available = neighbors
                
            if available:
                next_node = random.choice(available)
                trajectory.append(next_node)
                current_node = next_node
            else:
                break
    
    # Ensure the end point is near but not identical to real end point
    if real_end_node and trajectory:
        current_node = trajectory[-1]
        
        # Try to find a path to the real end node
        try:
            path_to_end = nx.shortest_path(G, current_node, real_end_node, weight='length')
            path_length = len(path_to_end) - 1  # Number of edges in path
            
            # Case 1: Too close to real end (closer than min_final_distance)
            if path_length < min_final_distance:
                # We need to move away slightly
                neighbors = list(G.neighbors(current_node))
                neighbors = [n for n in neighbors if n != real_end_node]
                
                if neighbors:
                    # Find neighbors that maintain desired distance
                    better_nodes = []
                    for n in neighbors:
                        try:
                            dist_to_end = len(nx.shortest_path(G, n, real_end_node)) - 1
                            if min_final_distance <= dist_to_end <= max_final_distance:
                                better_nodes.append(n)
                        except nx.NetworkXNoPath:
                            pass
                    
                    # Replace the last node with a better one
                    if better_nodes:
                        trajectory[-1] = random.choice(better_nodes)
                    elif neighbors:
                        # Just pick any neighbor if none meet the distance criteria
                        trajectory[-1] = random.choice(neighbors)
            
            # Case 2: Too far from real end (farther than max_final_distance)
            elif path_length > max_final_distance:
                # Move closer, but not too close
                # Get a node that's exactly min_final_distance away
                if len(path_to_end) > min_final_distance + 1:
                    target_idx = len(path_to_end) - min_final_distance - 1
                    if 0 < target_idx < len(path_to_end):
                        trajectory[-1] = path_to_end[target_idx]
                        
        except nx.NetworkXNoPath:
            pass
    
    return trajectory

def main():
    """Main function to generate and visualize trajectories"""
    # Example location in Ho Chi Minh City
    real_location = (10.772, 106.698)  # (lat, lon)
    
    # Parameters for trajectory generation - adjustable for testing
    global epsilon, similarity_factor
    epsilon = 0.1
    scale_factor = 12.0
    similarity_factor = 0.45  # Adjusted for better path balance
    
    # Other parameters
    num_points = 20
    randomness_factor = 0.4
    network_distance = 6000 

    print(f"Generating trajectory with {num_points} points and similarity factor {similarity_factor}")
    
    # Generate a real trajectory first - with larger area for more complex movements
    G, (nodes_gdf, edges_gdf) = get_road_network(real_location, distance=network_distance)
    real_point = Point(real_location[1], real_location[0])
    real_start_node = get_nearest_node(G, real_point)
    
    # Generate complex real trajectory
    print("Generating complex real trajectory...")
    real_trajectory_nodes = generate_complex_trajectory(G, real_start_node, length=num_points, 
                                                       randomness=randomness_factor)
    
    # Convert to points and edges
    real_traj_points, real_traj_edges = convert_nodes_to_points_and_edges(G, nodes_gdf, real_trajectory_nodes)
    real_traj = (real_traj_points, real_traj_edges)
    
    # Generate 4 fake privacy-preserving trajectories
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
        
        # Find nearest node to noisy start point
        noisy_point = Point(noisy_lon, noisy_lat)
        start_node = get_nearest_node(G, noisy_point)
        
        # Generate biased trajectory with end focus
        node_path = generate_biased_trajectory(
            G, nodes_gdf, real_trajectory_nodes, start_node,
            similarity_factor=similarity_factor, 
            length=num_points,
            end_focus=3
        )
        
        # Convert to points and edges
        trajectory_points, path_edges = convert_nodes_to_points_and_edges(G, nodes_gdf, node_path)
        fake_trajs.append((trajectory_points, path_edges))
    
    # Create visualization
    create_visualization(real_location, real_traj, fake_trajs)

# New function to create more complex trajectories
def generate_complex_trajectory(G, start_node, length=30, randomness=0.3, min_distance=300):
    """
    Generate a more complex and realistic trajectory with turns and varied movement
    
    Parameters:
        randomness: How often to take a random turn (0-1)
        min_distance: Minimum distance to move before considering a direction change
    """
    trajectory = [start_node]
    current_node = start_node
    
    # Try to get nodes that are far from start to use as destinations
    far_nodes = []
    try:
        # Get graph center
        center = nx.center(G)[0]
        # Get distances from center
        dist = nx.single_source_dijkstra_path_length(G, center, weight='length')
        # Get nodes that are at least min_distance away
        eligible_nodes = [node for node, d in dist.items() if d > min_distance and node != start_node]
        if eligible_nodes:
            # Pick several distant nodes as destinations
            far_nodes = random.sample(eligible_nodes, min(5, len(eligible_nodes)))
    except:
        # Fall back if the above fails
        far_nodes = []
    
    # If we couldn't get far nodes, just pick random ones
    if not far_nodes:
        nodes = list(G.nodes())
        if len(nodes) > 5 and start_node in nodes:
            nodes.remove(start_node)
            far_nodes = random.sample(nodes, min(5, len(nodes)))
    
    # Current destination node
    destination_idx = 0
    current_destination = far_nodes[destination_idx] if far_nodes else None
    
    # Previous direction to discourage zigzagging
    prev_direction = None
    
    for i in range(length-1):
        # Get neighboring nodes
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
            
        # If we have neighbors that haven't been visited already
        available = [n for n in neighbors if n not in trajectory]
        if not available and neighbors:
            available = neighbors
            
        if available:
            # With some probability, target our current destination
            if current_destination and np.random.random() > randomness and nx.has_path(G, current_node, current_destination):
                try:
                    # Find shortest path to the destination
                    path_to_dest = nx.shortest_path(G, current_node, current_destination, weight='length')
                    
                    # Check if the path has at least 2 nodes (current and next)
                    if len(path_to_dest) > 1:
                        next_node = path_to_dest[1]  # Take the first step toward destination
                    else:
                        # We're already at the destination, pick a new one
                        destination_idx = (destination_idx + 1) % len(far_nodes)
                        current_destination = far_nodes[destination_idx] if far_nodes else None
                        # Choose randomly for this step
                        next_node = random.choice(available)
                        
                    trajectory.append(next_node)
                    current_node = next_node
                    
                    # Check if we've reached the destination
                    if current_node == current_destination:
                        # Pick next destination
                        destination_idx = (destination_idx + 1) % len(far_nodes)
                        current_destination = far_nodes[destination_idx] if far_nodes else None
                except nx.NetworkXNoPath:
                    # Fall back to random selection if no path exists
                    next_node = random.choice(available)
                    trajectory.append(next_node)
                    current_node = next_node
            else:
                # Move randomly, but try to avoid immediate backtracking
                if prev_direction and len(available) > 1:
                    # Try to continue in same general direction when possible
                    weights = []
                    for n in available:
                        # Get coordinates to determine direction
                        curr_coords = (G.nodes[current_node]['x'], G.nodes[current_node]['y'])
                        prev_coords = (G.nodes[prev_direction]['x'], G.nodes[prev_direction]['y'])
                        next_coords = (G.nodes[n]['x'], G.nodes[n]['y'])
                        
                        # Calculate vectors
                        prev_vector = (curr_coords[0] - prev_coords[0], curr_coords[1] - prev_coords[1])
                        next_vector = (next_coords[0] - curr_coords[0], next_coords[1] - curr_coords[1])
                        
                        # Calculate dot product to determine if directions are similar
                        try:
                            dot = (prev_vector[0] * next_vector[0] + prev_vector[1] * next_vector[1])
                            # Higher weight for continuing in same direction
                            weight = max(0.1, dot)  # Ensure minimum weight
                        except:
                            weight = 1.0
                            
                        weights.append(weight)
                    
                    # Normalize weights
                    if sum(weights) > 0:
                        weights = [w/sum(weights) for w in weights]
                    else:
                        weights = None
                        
                    # Choose next node with calculated weights
                    try:
                        next_node = random.choices(available, weights=weights, k=1)[0]
                    except:
                        next_node = random.choice(available)
                else:
                    next_node = random.choice(available)
                    
                prev_direction = current_node  # Save previous node for directional bias
                trajectory.append(next_node)
                current_node = next_node
        else:
            break
            
    return trajectory

# Helper function to convert node paths to points and edges
def convert_nodes_to_points_and_edges(G, nodes_gdf, node_path):
    """Convert a node path to points and edges for visualization"""
    trajectory_points = []
    for node_id in node_path:
        node = nodes_gdf.loc[node_id]
        trajectory_points.append(Point(node.geometry.x, node.geometry.y))
    
    # Create a complete path with edges
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

def create_visualization(real_location, real_traj, fake_trajs):
    """Create a 2x2 grid visualization comparing real and fake trajectories"""
    # Create a single figure with 4 subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    axs = axs.flatten()
    
    # Colors for fake trajectories
    colors = ['blue', 'green', 'orange', 'purple']
    
    # Plot each fake trajectory in its own subplot
    for i, fake_traj in enumerate(fake_trajs):
        print(f"Creating subplot {i+1}/4...")
        ax = axs[i]
        fake_traj_points, fake_traj_edges = fake_traj
        
        # Create GeoDataFrames for real trajectory
        real_traj_gdf = gpd.GeoDataFrame(geometry=real_traj[0], crs="EPSG:4326").to_crs(epsg=3857)
        real_edge_lines = [LineString([u, v]) for u, v in real_traj[1]]
        real_edges_gdf = gpd.GeoDataFrame(geometry=real_edge_lines, crs="EPSG:4326").to_crs(epsg=3857)
        
        # Create GeoDataFrames for fake trajectory
        fake_traj_gdf = gpd.GeoDataFrame(geometry=fake_traj_points, crs="EPSG:4326").to_crs(epsg=3857)
        fake_edge_lines = [LineString([u, v]) for u, v in fake_traj_edges]
        fake_edges_gdf = gpd.GeoDataFrame(geometry=fake_edge_lines, crs="EPSG:4326").to_crs(epsg=3857)
        
        # Choose color for this fake trajectory
        color = colors[i]
        
        # Plot trajectories
        real_edges_gdf.plot(ax=ax, color='red', linewidth=3, alpha=0.8, label='Real')
        real_traj_gdf.iloc[[0]].plot(ax=ax, color='lime', markersize=60, alpha=1.0, marker='^', 
                                     edgecolor='black', linewidth=2, label='Real Start')
        real_traj_gdf.iloc[[-1]].plot(ax=ax, color='magenta', markersize=60, alpha=1.0, marker='s', 
                                      edgecolor='black', linewidth=2, label='Real End')
        
        fake_edges_gdf.plot(ax=ax, color=color, linewidth=2, alpha=0.6, label=f'Fake {i+1}')
        fake_traj_gdf.iloc[[0]].plot(ax=ax, color='cyan', markersize=50, alpha=1.0, marker='^', 
                                    edgecolor='black', linewidth=2, label=f'Fake Start {i+1}')
        fake_traj_gdf.iloc[[-1]].plot(ax=ax, color='yellow', markersize=50, alpha=1.0, marker='s', 
                                     edgecolor='black', linewidth=2, label=f'Fake End {i+1}')
        
        # Calculate boundaries
        minx_real, miny_real, maxx_real, maxy_real = real_edges_gdf.total_bounds
        minx_fake, miny_fake, maxx_fake, maxy_fake = fake_edges_gdf.total_bounds
        
        total_bounds = [
            min(minx_real, minx_fake),
            min(miny_real, miny_fake),
            max(maxx_real, maxx_fake),
            max(maxy_real, maxy_fake)
        ]
        
        # Set bounds with padding
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
        
        ax.set_title(f"Real vs. Fake Trajectory {i+1}", fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.axis('off')
    
    # Add title and save
    fig.suptitle(f"Road-Constrained Privacy-Preserving Trajectories (ε = {epsilon}, similarity = {similarity_factor:.1f})", 
                 fontsize=20, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save figure
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"result/{current_time}_four_trajectories_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print("Generated combined figure with 4 trajectory comparisons")

if __name__ == "__main__":
    for i in range(5):
        print(f"Run {i+1}")
        main()
        print(f"✅ Completed run {i+1}")
