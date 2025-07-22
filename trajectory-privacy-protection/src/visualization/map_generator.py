from folium import Map, PolyLine, Marker
import geopandas as gpd

def generate_map(real_trajectory, fake_trajectory, output_file='trajectory_map.html'):
    # Create a base map centered around the average location of the real trajectory
    if not real_trajectory:
        raise ValueError("Real trajectory is empty. Cannot generate map.")
    
    avg_lat = sum([point[1] for point in real_trajectory]) / len(real_trajectory)
    avg_lon = sum([point[0] for point in real_trajectory]) / len(real_trajectory)
    
    m = Map(location=[avg_lat, avg_lon], zoom_start=14)

    # Plot real trajectory
    if real_trajectory:
        real_line = PolyLine(locations=[(lat, lon) for lon, lat in real_trajectory], color='blue', weight=5, opacity=0.7, tooltip='Real Trajectory')
        real_line.add_to(m)

    # Plot fake trajectory
    if fake_trajectory:
        fake_line = PolyLine(locations=[(lat, lon) for lon, lat in fake_trajectory], color='red', weight=5, opacity=0.7, tooltip='Fake Trajectory')
        fake_line.add_to(m)

    # Add markers for start and end points
    if real_trajectory:
        start_marker = Marker(location=real_trajectory[0][1], popup='Start', icon=None)
        end_marker = Marker(location=real_trajectory[-1][1], popup='End', icon=None)
        start_marker.add_to(m)
        end_marker.add_to(m)

    # Save the map to an HTML file
    m.save(output_file)
    print(f"Map saved to {output_file}")