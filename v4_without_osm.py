import numpy as np
import math
import folium # For visualization
from shapely.geometry import Point, Polygon, LineString # For geometric operations
# In a real scenario, you'd use a library like GeoPandas for easier handling of geospatial data
# and OSMnx or similar for interacting with OpenStreetMap data for realism checks.

# --- Configuration & Helper Functions ---
PRIVACY_EPSILON = 0.1 # Smaller epsilon = more privacy
MAX_REALISTIC_ATTEMPTS = 10
SEARCH_RADIUS_REALISTIC = 50 # meters, for finding nearby realistic points
MAX_PLAUSIBLE_SPEED_KMH = 80 # For path coherence (example)

# Define a sample QoS Region (Polygon) - e.g., a rectangular area
# Coordinates: (longitude, latitude)
QOS_REGION_COORDS = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]
QOS_POLYGON = Polygon(QOS_REGION_COORDS)

# Define sample Start/End Regions (Polygons)
START_REGION_COORDS = [( -0.001, -0.001), (-0.001, 0.001), (0.001, 0.001), (0.001, -0.001)]
END_REGION_COORDS = [(0.009, 0.009), (0.009, 0.011), (0.011, 0.011), (0.011, 0.009)]
START_POLYGON = Polygon(START_REGION_COORDS)
END_POLYGON = Polygon(END_REGION_COORDS)


# Placeholder for map data - in reality, this would query a spatial database/API
# For this example, we'll define a few "forbidden" rectangular zones (buildings, water)
FORBIDDEN_ZONES_COORDS = [
    [(0.002, 0.002), (0.002, 0.003), (0.003, 0.003), (0.003, 0.002)], # Building 1
    [(0.007, 0.006), (0.007, 0.007), (0.008, 0.007), (0.008, 0.006)], # Water Body 1
]
FORBIDDEN_POLYGONS = [Polygon(coords) for coords in FORBIDDEN_ZONES_COORDS]

def planar_laplace_noise(epsilon):
    """Generates 2D Laplace noise for Geo-Indistinguishability."""
    # For simplicity, using a variant where scale = 1/epsilon.
    # A more standard way involves inverse CDF sampling for r and uniform for theta.
    # Here, we simplify by drawing two 1D Laplace samples.
    # This is a simplification; true planar Laplace is more complex.
    # A common approximation is to use two independent 1D Laplace noises for x and y.
    # Scale parameter b = sensitivity / epsilon. Assuming sensitivity = 1 (for normalized coords or small distances).
    scale = 1.0 / epsilon
    u1 = np.random.uniform(-0.5, 0.5)
    u2 = np.random.uniform(-0.5, 0.5)
    dx = scale * np.sign(u1) * np.log(1 - 2 * np.abs(u1))
    dy = scale * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
    # Scale down noise significantly for lat/lon degrees for this example
    # This scaling factor is arbitrary and needs to be tuned based on coordinate system and desired noise magnitude
    dx *= 0.0001
    dy *= 0.0001
    return dx, dy

def project_to_qos_boundary(point_coords, qos_polygon):
    """Projects a point to the nearest point on the QoS polygon's boundary if it's outside."""
    point = Point(point_coords)
    if not qos_polygon.contains(point):
        # Shapely's `interpolate` and `project` can find the nearest point on the boundary
        # Project the point onto the boundary line
        projected_point = qos_polygon.boundary.interpolate(qos_polygon.boundary.project(point))
        return (projected_point.x, projected_point.y)
    return point_coords # Already inside

def is_location_realistic(point_coords, forbidden_polygons):
    """Checks if a point is in a realistic location (not in forbidden zones)."""
    point = Point(point_coords)
    for zone in forbidden_polygons:
        if zone.contains(point):
            return False
    # Add more checks: e.g., on a road, not in water (requires detailed map data)
    return True # Assume realistic if not in explicitly forbidden zones for this example

def find_nearby_realistic_point(center_coords, forbidden_polygons, search_radius_deg, qos_polygon):
    """
    Tries to find a realistic point near the center_coords by random sampling in a radius.
    search_radius_deg should be small, e.g., corresponding to meters.
    This is a simplified version.
    """
    center_x, center_y = center_coords
    for _ in range(MAX_REALISTIC_ATTEMPTS // 2) : # Try a few times
        # Generate a random point within a small circle
        angle = np.random.uniform(0, 2 * np.pi)
        # Convert search_radius (meters) to degrees (approximate and naive for example)
        # 1 degree lat ~ 111 km. 1 degree lon ~ 111 km * cos(lat)
        # This is a very rough approximation for small distances
        radius_deg_x = search_radius_deg / (111000 * math.cos(math.radians(center_y)))
        radius_deg_y = search_radius_deg / 111000

        r_x = np.random.uniform(0, radius_deg_x)
        r_y = np.random.uniform(0, radius_deg_y)

        offset_x = r_x * np.cos(angle)
        offset_y = r_y * np.sin(angle) # Using r_y for y offset scale

        candidate_pt_coords = (center_x + offset_x, center_y + offset_y)
        candidate_pt = Point(candidate_pt_coords)

        if qos_polygon.contains(candidate_pt) and is_location_realistic(candidate_pt_coords, forbidden_polygons):
            return candidate_pt_coords
    return None # Could not find a realistic point nearby quickly

def get_random_point_in_polygon(polygon):
    """Gets a random point within a polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return (random_point.x, random_point.y)

# --- R-GITS Algorithm Implementation ---
def r_gits_generate_fake_point(true_loc_coords, epsilon, qos_polygon, forbidden_polygons,
                               is_start=False, start_polygon=None,
                               is_end=False, end_polygon=None,
                               last_fake_point_coords=None, last_timestamp=None, current_timestamp=None):
    """
    Generates one fake point using the R-GITS logic.
    true_loc_coords: (lon, lat)
    """
    current_true_for_geo_i = true_loc_coords

    # 1. Start Point Handling
    if is_start and start_polygon:
        current_true_for_geo_i = get_random_point_in_polygon(start_polygon)

    # 2. Generate Candidate Fake Point (Geo-Indistinguishability)
    dx, dy = planar_laplace_noise(epsilon)
    candidate_fake_coords = (current_true_for_geo_i[0] + dx, current_true_for_geo_i[1] + dy)

    # 3. QoS Clamping
    candidate_fake_coords = project_to_qos_boundary(candidate_fake_coords, qos_polygon)

    # 4. Realistic Point Verification and Adjustment
    final_fake_coords = candidate_fake_coords
    if not is_location_realistic(candidate_fake_coords, forbidden_polygons):
        realistic_nearby = find_nearby_realistic_point(candidate_fake_coords, forbidden_polygons, SEARCH_RADIUS_REALISTIC, qos_polygon)
        if realistic_nearby:
            final_fake_coords = realistic_nearby
        else:
            # Fallback: If no realistic point found nearby, use the QoS-clamped point.
            # More advanced fallbacks could be: closest realistic point on QoS boundary, or even previous fake point if sensible.
            # For this example, we proceed with the QoS-clamped if no quick realistic alternative is found.
            # Or, a stricter approach: if not realistic, consider it a failure for this point.
            # Here, we'll just use candidate_fake_coords, assuming it might be acceptable if narrowly missing.
            # A better fallback would be to find the *closest* realistic point within QoS.
            print(f"Warning: Could not find a clearly realistic point for {candidate_fake_coords}, using QoS-clamped version.")
            final_fake_coords = candidate_fake_coords # Defaulting to QoS clamped if no better found

    # 5. Path Coherence (Simplified Check - more rigorous check would involve speed)
    # This is a placeholder for a more advanced check.
    # For example, if last_fake_point_coords and timestamps are available:
    # dist = great_circle(last_fake_point_coords, final_fake_coords).km
    # time_delta_hours = (current_timestamp - last_timestamp).total_seconds() / 3600
    # speed = dist / time_delta_hours if time_delta_hours > 0 else float('inf')
    # if speed > MAX_PLAUSIBLE_SPEED_KMH:
    #     print(f"Warning: High speed detected ({speed} km/h). Path coherence might be an issue.")
        # Corrective action would be complex: e.g., move point closer to last_fake_point_coords
        # along the vector from last_fake_point_coords to final_fake_coords, while staying in QoS and realistic.

    # 6. End Point Handling
    if is_end and end_polygon:
        # Re-obfuscate based on the end region
        true_end_for_geo_i = get_random_point_in_polygon(end_polygon)
        dx_end, dy_end = planar_laplace_noise(epsilon)
        candidate_end_fake_coords = (true_end_for_geo_i[0] + dx_end, true_end_for_geo_i[1] + dy_end)
        candidate_end_fake_coords = project_to_qos_boundary(candidate_end_fake_coords, qos_polygon)

        if not is_location_realistic(candidate_end_fake_coords, forbidden_polygons):
            realistic_nearby_end = find_nearby_realistic_point(candidate_end_fake_coords, forbidden_polygons, SEARCH_RADIUS_REALISTIC, qos_polygon)
            if realistic_nearby_end:
                final_fake_coords = realistic_nearby_end
            else:
                final_fake_coords = candidate_end_fake_coords # Fallback
                print(f"Warning: Could not find a clearly realistic END point for {candidate_end_fake_coords}, using QoS-clamped version.")
        else:
            final_fake_coords = candidate_end_fake_coords


    return final_fake_coords


# --- Simulation & Visualization ---
if __name__ == '__main__':
    # Sample Real Trajectory (list of (lon, lat) tuples)
    real_trajectory_coords = [
        (0.0005, 0.0005), # Start
        (0.0015, 0.0010),
        (0.0025, 0.0020),
        (0.0035, 0.0035),
        (0.0050, 0.0045),
        (0.0060, 0.0055),
        (0.0075, 0.0070),
        (0.0085, 0.0080),
        (0.0095, 0.0095)  # End
    ]

    fake_trajectory_coords = []
    last_fk_point = None

    print("Processing trajectory...")
    for i, true_loc in enumerate(real_trajectory_coords):
        is_start_pt = (i == 0)
        is_end_pt = (i == len(real_trajectory_coords) - 1)

        # Timestamps are not used in this simplified path coherence, but would be needed for speed checks
        fake_pt = r_gits_generate_fake_point(
            true_loc, PRIVACY_EPSILON, QOS_POLYGON, FORBIDDEN_POLYGONS,
            is_start=is_start_pt, start_polygon=START_POLYGON,
            is_end=is_end_pt, end_polygon=END_POLYGON,
            last_fake_point_coords=last_fk_point
        )
        fake_trajectory_coords.append(fake_pt)
        last_fk_point = fake_pt
        print(f"True: {true_loc} -> Fake: {fake_pt}")

    # --- Visualization with Folium ---
    # Calculate map center (average of QoS region for simplicity)
    map_center_lon = sum(c[0] for c in QOS_REGION_COORDS) / len(QOS_REGION_COORDS)
    map_center_lat = sum(c[1] for c in QOS_REGION_COORDS) / len(QOS_REGION_COORDS)
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=15, tiles="OpenStreetMap")

    # Helper to convert (lon, lat) to (lat, lon) for Folium
    def to_lat_lon(coords_list):
        return [(c[1], c[0]) for c in coords_list]

    # Plot QoS Region
    folium.Polygon(
        locations=to_lat_lon(QOS_REGION_COORDS),
        color="blue", fill=True, fill_color="blue", fill_opacity=0.1,
        tooltip="QoS Region"
    ).add_to(m)

    # Plot Start Region
    folium.Polygon(
        locations=to_lat_lon(START_REGION_COORDS),
        color="green", fill=True, fill_color="green", fill_opacity=0.2,
        tooltip="Start Region"
    ).add_to(m)

    # Plot End Region
    folium.Polygon(
        locations=to_lat_lon(END_REGION_COORDS),
        color="purple", fill=True, fill_color="purple", fill_opacity=0.2,
        tooltip="End Region"
    ).add_to(m)


    # Plot Forbidden Zones
    for zone_coords in FORBIDDEN_ZONES_COORDS:
        folium.Polygon(
            locations=to_lat_lon(zone_coords),
            color="gray", fill=True, fill_color="gray", fill_opacity=0.5,
            tooltip="Forbidden Zone"
        ).add_to(m)

    # Plot Real Trajectory
    if real_trajectory_coords:
        folium.PolyLine(to_lat_lon(real_trajectory_coords), color="red", weight=2.5, opacity=1, tooltip="Real Trajectory").add_to(m)
        for i, p_coords in enumerate(real_trajectory_coords):
            folium.CircleMarker(
                location=(p_coords[1], p_coords[0]), radius=5, color="red", fill=True, fill_color="darkred",
                tooltip=f"Real Point {i+1}: ({p_coords[0]:.4f}, {p_coords[1]:.4f})"
            ).add_to(m)

    # Plot Fake Trajectory
    if fake_trajectory_coords:
        folium.PolyLine(to_lat_lon(fake_trajectory_coords), color="blue", weight=2.5, opacity=0.7, dash_array='5, 5', tooltip="Fake Trajectory").add_to(m)
        for i, p_coords in enumerate(fake_trajectory_coords):
            folium.CircleMarker(
                location=(p_coords[1], p_coords[0]), radius=5, color="blue", fill=True, fill_color="darkblue",
                tooltip=f"Fake Point {i+1}: ({p_coords[0]:.4f}, {p_coords[1]:.4f})"
            ).add_to(m)

    # Save map to an HTML file
    map_file = "trajectory_privacy_map.html"
    m.save(map_file)
    print(f"\nMap saved to {map_file}")
    print("Note: Realism checks in this example are basic. True realism requires rich map data (e.g., OpenStreetMap).")
    print("Planar Laplace noise here is a simplified version for x,y. Consider a more rigorous implementation for polar coordinates if needed.")
    print("Path coherence is also very basic; speed and routeability checks would enhance it.")