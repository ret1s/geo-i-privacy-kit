from folium import Map, Marker, PolyLine
from shapely.geometry import Point

def plot_trajectory_on_map(real_trajectory, fake_trajectory, map_file='trajectory_map.html'):
    # Create a base map
    if not real_trajectory and not fake_trajectory:
        print("No trajectories to plot.")
        return

    # Calculate the center of the map based on the trajectories
    latitudes = []
    longitudes = []

    if real_trajectory:
        latitudes.extend([point[1] for point in real_trajectory])
        longitudes.extend([point[0] for point in real_trajectory])

    if fake_trajectory:
        latitudes.extend([point[1] for point in fake_trajectory])
        longitudes.extend([point[0] for point in fake_trajectory])

    center_lat = sum(latitudes) / len(latitudes) if latitudes else 0
    center_lon = sum(longitudes) / len(longitudes) if longitudes else 0

    # Initialize the map
    m = Map(location=[center_lat, center_lon], zoom_start=14)

    # Plot real trajectory
    if real_trajectory:
        real_line = PolyLine(locations=[(point[1], point[0]) for point in real_trajectory], color='blue', weight=2.5, opacity=1)
        real_line.add_to(m)
        for point in real_trajectory:
            Marker(location=(point[1], point[0]), popup='Real Point', icon=None).add_to(m)

    # Plot fake trajectory
    if fake_trajectory:
        fake_line = PolyLine(locations=[(point[1], point[0]) for point in fake_trajectory], color='red', weight=2.5, opacity=1)
        fake_line.add_to(m)
        for point in fake_trajectory:
            Marker(location=(point[1], point[0]), popup='Fake Point', icon=None).add_to(m)

    # Save the map to an HTML file
    m.save(map_file)
    print(f"Map saved to {map_file}")