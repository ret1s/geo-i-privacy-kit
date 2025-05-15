import numpy as np
import math
import folium # For visualization
from shapely.geometry import Point, Polygon, LineString # For geometric operations
import random # For selecting random points in polygons

# --- Configuration & Helper Functions ---
PRIVACY_EPSILON = 0.5 # Smaller epsilon = more privacy, larger epsilon = less noise
MAX_REALISTIC_ATTEMPTS = 15
SEARCH_RADIUS_REALISTIC_METERS = 30 # meters, for finding nearby realistic points
MAX_PLAUSIBLE_SPEED_KMH = 60 # For path coherence (example)

# --- Ho Chi Minh City Coordinates ---
# Approximate center for example: around District 1
HCMC_CENTER_LAT = 10.7769  # Latitude
HCMC_CENTER_LON = 106.7009 # Longitude

# Define a sample QoS Region (Polygon) in HCMC - approx 1km x 1km square
# (longitude, latitude)
offset_qos = 0.005 # Roughly 500m in degrees
QOS_REGION_COORDS = [
    (HCMC_CENTER_LON - offset_qos, HCMC_CENTER_LAT - offset_qos),
    (HCMC_CENTER_LON - offset_qos, HCMC_CENTER_LAT + offset_qos),
    (HCMC_CENTER_LON + offset_qos, HCMC_CENTER_LAT + offset_qos),
    (HCMC_CENTER_LON + offset_qos, HCMC_CENTER_LAT - offset_qos),
]
QOS_POLYGON = Polygon(QOS_REGION_COORDS)

# Define sample Start/End Regions (Polygons) within HCMC
offset_start_end = 0.001 # Roughly 100m
START_REGION_COORDS = [
    (HCMC_CENTER_LON - offset_qos + 0.0005, HCMC_CENTER_LAT - offset_qos + 0.0005),
    (HCMC_CENTER_LON - offset_qos + 0.0005, HCMC_CENTER_LAT - offset_qos + 0.0015),
    (HCMC_CENTER_LON - offset_qos + 0.0015, HCMC_CENTER_LAT - offset_qos + 0.0015),
    (HCMC_CENTER_LON - offset_qos + 0.0015, HCMC_CENTER_LAT - offset_qos + 0.0005),
]
END_REGION_COORDS = [
    (HCMC_CENTER_LON + offset_qos - 0.0015, HCMC_CENTER_LAT + offset_qos - 0.0015),
    (HCMC_CENTER_LON + offset_qos - 0.0015, HCMC_CENTER_LAT + offset_qos - 0.0005),
    (HCMC_CENTER_LON + offset_qos - 0.0005, HCMC_CENTER_LAT + offset_qos - 0.0005),
    (HCMC_CENTER_LON + offset_qos - 0.0005, HCMC_CENTER_LAT + offset_qos - 0.0015),
]
START_POLYGON = Polygon(START_REGION_COORDS)
END_POLYGON = Polygon(END_REGION_COORDS)

# Placeholder for map data - manually defined "forbidden" rectangular zones in HCMC
# These would ideally be replaced by actual building/water data from OSM via OSMnx
FORBIDDEN_ZONES_COORDS = [
    [(HCMC_CENTER_LON + 0.001, HCMC_CENTER_LAT + 0.001),
     (HCMC_CENTER_LON + 0.001, HCMC_CENTER_LAT + 0.0015),
     (HCMC_CENTER_LON + 0.0015, HCMC_CENTER_LAT + 0.0015),
     (HCMC_CENTER_LON + 0.0015, HCMC_CENTER_LAT + 0.001)], # "Building 1"
    [(HCMC_CENTER_LON - 0.002, HCMC_CENTER_LAT - 0.002),
     (HCMC_CENTER_LON - 0.002, HCMC_CENTER_LAT - 0.0015),
     (HCMC_CENTER_LON - 0.0015, HCMC_CENTER_LAT - 0.0015),
     (HCMC_CENTER_LON - 0.0015, HCMC_CENTER_LAT - 0.002)], # "Water Body 1"
]
FORBIDDEN_POLYGONS = [Polygon(coords) for coords in FORBIDDEN_ZONES_COORDS]

# --- OSMnx Integration Placeholder ---
# To use OSMnx for advanced realism, you would uncomment and complete this section.
# You'll need to install osmnx: pip install osmnx
# import osmnx as ox
# G_map_hcmc = None
# buildings_hcmc = None
# water_hcmc = None

# def initialize_osm_data_hcmc():
#     global G_map_hcmc, buildings_hcmc, water_hcmc
#     # Define the bounding box for the area of interest in HCMC
#     # Example: slightly larger than our QoS region
#     north, south = HCMC_CENTER_LAT + offset_qos + 0.005, HCMC_CENTER_LAT - offset_qos - 0.005
#     east, west = HCMC_CENTER_LON + offset_qos + 0.005, HCMC_CENTER_LON - offset_qos - 0.005
#     try:
#         print("Attempting to download OSM data for HCMC area (this may take a moment)...")
#         # Download street network (e.g., for checking road proximity)
#         G_map_hcmc = ox.graph_from_bbox(north, south, east, west, network_type='drive_service', simplify=True, retain_all=True)
#
#         # Download building footprints
#         tags_building = {'building': True}
#         buildings_hcmc = ox.features_from_bbox(north, south, east, west, tags=tags_building)
#         print(f"Loaded {len(buildings_hcmc)} building features from OSM.")
#
#         # Download water features
#         tags_water = {'natural': ['water', 'bay'], 'waterway': True, 'landuse': 'reservoir'}
#         water_hcmc = ox.features_from_bbox(north, south, east, west, tags=tags_water)
#         print(f"Loaded {len(water_hcmc)} water features from OSM.")
#
#         print("OSM data initialized.")
#     except Exception as e:
#         print(f"Could not load OSM data: {e}. Realism checks will be limited.")
#         G_map_hcmc, buildings_hcmc, water_hcmc = None, None, None

# Call this once at the start if you want to use OSMnx
# initialize_osm_data_hcmc()


def planar_laplace_noise(epsilon, sensitivity=1):
    """
    Generates 2D Laplace noise. For simplicity, draws two independent 1D Laplace samples.
    A more standard planar Laplace mechanism involves sampling radius and angle.
    Sensitivity here is a scaling factor for the noise based on the domain.
    For lat/lon, sensitivity would be small (e.g., max distance error allowed / epsilon).
    Let's use a simpler scale factor for now.
    """
    # Scale parameter b = sensitivity / epsilon.
    # A practical approach for geographic coords: scale defines approx. meters of obfuscation.
    # Example: if epsilon = 0.1, and we want noise around 100m, scale = 100.
    # This is not strictly Geo-I's definition but a pragmatic way to control noise magnitude.
    # For Geo-I, sensitivity is often 1 for normalized distances, or the max distance between points.
    # The scaling factor below is heuristic and needs to be chosen based on desired physical obfuscation.
    # 1 degree lat ~ 111 km. 0.001 degree ~ 111 meters.
    # Let's aim for noise in the order of 0.0001 to 0.001 degrees.
    # scale = (sensitivity_in_degrees / epsilon)
    # If sensitivity is 0.001 degrees, scale = 0.001 / epsilon
    effective_scale = 0.001 / epsilon # Heuristic: for epsilon=0.1, noise is ~0.01 deg; for e=1, ~0.001 deg

    u1 = np.random.uniform(-0.5, 0.5)
    u2 = np.random.uniform(-0.5, 0.5)
    # Laplace L(0, b) can be generated as b * sgn(U-0.5) * ln(1-2|U-0.5|)
    # However, many libraries use scale = 1/lambda, where lambda = epsilon/sensitivity. So b = sensitivity/epsilon
    dx = effective_scale * np.sign(u1) * np.log(1 - 2 * np.abs(u1))
    dy = effective_scale * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
    return dx, dy

def project_to_qos_boundary(point_coords, qos_polygon):
    point = Point(point_coords)
    if not qos_polygon.contains(point):
        projected_point = qos_polygon.boundary.interpolate(qos_polygon.boundary.project(point))
        return (projected_point.x, projected_point.y)
    return point_coords

def is_location_realistic(point_coords, qos_polygon, forbidden_polygons_manual):
    """
    Checks if a point is in a realistic location.
    Enhanced version would use OSM data (G_map_hcmc, buildings_hcmc, water_hcmc).
    """
    point = Point(point_coords)

    # Basic check: Must be within QoS (though usually clamped before this)
    if not qos_polygon.contains(point):
        return False # Should not happen if clamped correctly

    # Manual Forbidden Zones (simple check)
    for zone in forbidden_polygons_manual:
        if zone.contains(point):
            # print(f"Point {point_coords} rejected: in manual forbidden zone.")
            return False

    # --- Advanced OSMnx based checks (conceptual) ---
    # global buildings_hcmc, water_hcmc # G_map_hcmc
    # if buildings_hcmc is not None and not buildings_hcmc.empty:
    #     # Check if point is within any building polygon. Requires GeoDataFrame.
    #     # This check can be slow if not optimized (e.g., using spatial index).
    #     try:
    #         if buildings_hcmc.geometry.contains(point).any():
    #             print(f"Point {point_coords} rejected: inside OSM building.")
    #             return False
    #     except Exception as e:
    #         print(f"Error checking OSM buildings: {e}") # Catch potential errors with empty GDFs etc.
    #
    # if water_hcmc is not None and not water_hcmc.empty:
    #     try:
    #         if water_hcmc.geometry.contains(point).any():
    #             print(f"Point {point_coords} rejected: inside OSM water body.")
    #             return False
    #     except Exception as e:
    #         print(f"Error checking OSM water: {e}")

    # (Optional) Proximity to road/path network (more complex)
    # if G_map_hcmc is not None:
    #     try:
    #         # Find nearest edge. Note: X, Y order for nearest_edges is lon, lat.
    #         _, dist = ox.distance.nearest_edges(G_map_hcmc, X=point.x, Y=point.y, return_dist=True)
    #         # Assuming G_map_hcmc is projected to meters for meaningful dist.
    #         # If not projected, dist might be in degrees or incorrect.
    #         # For unprojected graph, ox.distance.great_circle_vec might be better.
    #         # This part needs careful handling of projections for accurate distance.
    #         # Let's assume a threshold in degrees if not projected, e.g., 0.0005 deg ~ 50m
    #         # if dist > 0.0005: # Example threshold
    #         #     print(f"Point {point_coords} rejected: too far from OSM road network (dist: {dist}).")
    #         #     return False
    #     except Exception as e:
    #         print(f"Error checking OSM road proximity: {e}")
    return True

def find_nearby_realistic_point(center_coords, qos_polygon, forbidden_polygons_manual, search_radius_m):
    center_lon, center_lat = center_coords
    # Convert search_radius_m to approximate degrees
    # 1 deg lat ~ 111km. search_radius_m / (111000)
    # 1 deg lon ~ 111km * cos(lat). search_radius_m / (111000 * cos(lat_rad))
    lat_rad = math.radians(center_lat)
    radius_deg_lat = search_radius_m / 111000.0
    radius_deg_lon = search_radius_m / (111000.0 * math.cos(lat_rad) + 1e-9) # Add epsilon to avoid division by zero near poles

    for _ in range(MAX_REALISTIC_ATTEMPTS // 2):
        angle = np.random.uniform(0, 2 * np.pi)
        # Sample within an ellipse to approximate circular region in meters
        r_lon = np.random.uniform(0, radius_deg_lon)
        r_lat = np.random.uniform(0, radius_deg_lat)

        offset_lon = r_lon * math.cos(angle)
        offset_lat = r_lat * math.sin(angle)

        candidate_pt_coords = (center_lon + offset_lon, center_lat + offset_lat)

        if qos_polygon.contains(Point(candidate_pt_coords)) and \
           is_location_realistic(candidate_pt_coords, qos_polygon, forbidden_polygons_manual):
            return candidate_pt_coords
    return None

def get_random_point_in_polygon(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(random_point):
            return (random_point.x, random_point.y)

def r_gits_generate_fake_point(true_loc_coords, epsilon, qos_polygon, forbidden_polygons_manual,
                               is_start=False, start_polygon=None,
                               is_end=False, end_polygon=None,
                               last_fake_point_coords=None): # Timestamps needed for full coherence
    current_true_for_geo_i = true_loc_coords

    if is_start and start_polygon:
        current_true_for_geo_i = get_random_point_in_polygon(start_polygon)

    dx, dy = planar_laplace_noise(epsilon)
    candidate_fake_coords = (current_true_for_geo_i[0] + dx, current_true_for_geo_i[1] + dy)
    candidate_fake_coords = project_to_qos_boundary(candidate_fake_coords, qos_polygon)

    final_fake_coords = candidate_fake_coords
    if not is_location_realistic(candidate_fake_coords, qos_polygon, forbidden_polygons_manual):
        realistic_nearby = find_nearby_realistic_point(candidate_fake_coords, qos_polygon, forbidden_polygons_manual, SEARCH_RADIUS_REALISTIC_METERS)
        if realistic_nearby:
            final_fake_coords = realistic_nearby
        else:
            # Fallback: Use the QoS-clamped point if no realistic nearby point is found quickly.
            # This is a simplification; a better fallback might be the *closest* realistic point.
            # print(f"Warning: Could not find a clearly realistic point for {candidate_fake_coords} after trying. Using QoS-clamped.")
            pass # final_fake_coords is already candidate_fake_coords

    if is_end and end_polygon:
        true_end_for_geo_i = get_random_point_in_polygon(end_polygon)
        dx_end, dy_end = planar_laplace_noise(epsilon)
        candidate_end_fake_coords = (true_end_for_geo_i[0] + dx_end, true_end_for_geo_i[1] + dy_end)
        candidate_end_fake_coords = project_to_qos_boundary(candidate_end_fake_coords, qos_polygon)

        if is_location_realistic(candidate_end_fake_coords, qos_polygon, forbidden_polygons_manual):
            final_fake_coords = candidate_end_fake_coords
        else:
            realistic_nearby_end = find_nearby_realistic_point(candidate_end_fake_coords, qos_polygon, forbidden_polygons_manual, SEARCH_RADIUS_REALISTIC_METERS)
            if realistic_nearby_end:
                final_fake_coords = realistic_nearby_end
            else: # Fallback for end point
                final_fake_coords = candidate_end_fake_coords
                # print(f"Warning: Could not find a clearly realistic END point for {candidate_end_fake_coords}. Using QoS-clamped.")
    return final_fake_coords

# --- Simulation & Visualization ---
if __name__ == '__main__':
    # Sample Real Trajectory in HCMC (list of (lon, lat) tuples)
    # Moving roughly from SW of center to NE of center within QoS general area
    real_trajectory_coords = [
        (HCMC_CENTER_LON - 0.004, HCMC_CENTER_LAT - 0.004), # Start
        (HCMC_CENTER_LON - 0.003, HCMC_CENTER_LAT - 0.002),
        (HCMC_CENTER_LON - 0.001, HCMC_CENTER_LAT - 0.000),
        (HCMC_CENTER_LON + 0.001, HCMC_CENTER_LAT + 0.001),
        (HCMC_CENTER_LON + 0.002, HCMC_CENTER_LAT + 0.003),
        (HCMC_CENTER_LON + 0.004, HCMC_CENTER_LAT + 0.004)  # End
    ]

    fake_trajectory_coords = []
    last_fk_point = None

    print("Processing trajectory...")
    for i, true_loc in enumerate(real_trajectory_coords):
        is_start_pt = (i == 0)
        is_end_pt = (i == len(real_trajectory_coords) - 1)

        fake_pt = r_gits_generate_fake_point(
            true_loc, PRIVACY_EPSILON, QOS_POLYGON, FORBIDDEN_POLYGONS,
            is_start=is_start_pt, start_polygon=START_POLYGON,
            is_end=is_end_pt, end_polygon=END_POLYGON,
            last_fake_point_coords=last_fk_point
        )
        fake_trajectory_coords.append(fake_pt)
        last_fk_point = fake_pt
        print(f"True: ({true_loc[0]:.5f}, {true_loc[1]:.5f}) -> Fake: ({fake_pt[0]:.5f}, {fake_pt[1]:.5f})")

    # --- Visualization with Folium on OpenStreetMap ---
    # Map centered on HCMC_CENTER_LAT, HCMC_CENTER_LON
    m = folium.Map(location=[HCMC_CENTER_LAT, HCMC_CENTER_LON], zoom_start=15, tiles="OpenStreetMap")

    def to_lat_lon_folium(coords_list_lon_lat): # Folium expects (lat, lon)
        return [(c[1], c[0]) for c in coords_list_lon_lat]

    folium.Polygon(
        locations=to_lat_lon_folium(QOS_REGION_COORDS),
        color="blue", fill=True, fill_color="blue", fill_opacity=0.1,
        tooltip="QoS Region"
    ).add_to(m)
    folium.Polygon(
        locations=to_lat_lon_folium(START_REGION_COORDS),
        color="green", fill=True, fill_color="green", fill_opacity=0.2,
        tooltip="Plausible Start Region"
    ).add_to(m)
    folium.Polygon(
        locations=to_lat_lon_folium(END_REGION_COORDS),
        color="purple", fill=True, fill_color="purple", fill_opacity=0.2,
        tooltip="Plausible End Region"
    ).add_to(m)
    for zone_coords in FORBIDDEN_ZONES_COORDS:
        folium.Polygon(
            locations=to_lat_lon_folium(zone_coords),
            color="black", fill=True, fill_color="dimgray", fill_opacity=0.4,
            tooltip="Forbidden Zone (Manual)"
        ).add_to(m)

    if real_trajectory_coords:
        folium.PolyLine(to_lat_lon_folium(real_trajectory_coords), color="red", weight=3, opacity=0.8, tooltip="Real Trajectory").add_to(m)
        for i, p_coords in enumerate(real_trajectory_coords):
            folium.CircleMarker(
                location=(p_coords[1], p_coords[0]), radius=6, color="red", fill=True, fill_color="darkred",
                tooltip=f"Real Pt {i+1}: ({p_coords[0]:.4f}, {p_coords[1]:.4f})"
            ).add_to(m)

    if fake_trajectory_coords:
        folium.PolyLine(to_lat_lon_folium(fake_trajectory_coords), color="dodgerblue", weight=3, opacity=0.8, dash_array='10, 5', tooltip="Fake Trajectory (R-GITS)").add_to(m)
        for i, p_coords in enumerate(fake_trajectory_coords):
            folium.CircleMarker(
                location=(p_coords[1], p_coords[0]), radius=6, color="blue", fill=True, fill_color="royalblue",
                tooltip=f"Fake Pt {i+1}: ({p_coords[0]:.4f}, {p_coords[1]:.4f})"
            ).add_to(m)

    map_file = "hcmc_trajectory_privacy_map.html"
    m.save(map_file)
    print(f"\nMap saved to {map_file}")
    print("Open this HTML file in a browser to see the trajectories on an OpenStreetMap of Ho Chi Minh City.")
    print("\n--- Notes for Advanced Realism (OSMnx) ---")
    print("1. To use actual OSM data for realism (buildings, water, roads):")
    print("   - Install OSMnx: pip install osmnx geopandas")
    print("   - Uncomment the 'OSMnx Integration Placeholder' section.")
    print("   - Uncomment and complete the OSM-based checks within 'is_location_realistic'.")
    print("   - Be aware that downloading OSM data (initialize_osm_data_hcmc) requires an internet connection and can take time.")
    print("   - OSMnx geometric operations are more accurate on projected graphs.")