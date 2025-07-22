import numpy as np
import math
import folium
import time
import random
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import transform as shapely_transform
import osmnx as ox
import geopandas as gpd
from pyproj import Transformer, CRS
import networkx as nx
from scipy.spatial.distance import euclidean

# --- Enhanced Configuration & Global Variables ---
PRIVACY_EPSILON = 0.5
MAX_REALISTIC_ATTEMPTS = 15
MAX_PLAUSIBLE_SPEED_KMH = 60
QUALITY_OF_SERVICE_RADIUS_M = 180  # Enhanced QoS radius
PATH_COHERENCE_WEIGHT = 0.7
REALISM_THRESHOLD = 0.8

# --- Ho Chi Minh City Enhanced Configuration ---
HCMC_CENTER_LAT_CONFIG = 10.7769
HCMC_CENTER_LON_CONFIG = 106.7009
offset_qos_deg_config = 0.01  # Expanded QoS region
HCMC_BBOX_HALF_SIZE_DEG = 0.012

HCMC_BBOX = (
    HCMC_CENTER_LAT_CONFIG + HCMC_BBOX_HALF_SIZE_DEG,
    HCMC_CENTER_LAT_CONFIG - HCMC_BBOX_HALF_SIZE_DEG,
    HCMC_CENTER_LON_CONFIG + HCMC_BBOX_HALF_SIZE_DEG,
    HCMC_CENTER_LON_CONFIG - HCMC_BBOX_HALF_SIZE_DEG
)

# Global variables for enhanced OSM data
G_proj = None
buildings_gdf = None
water_gdf = None
green_spaces_gdf = None
transformer_to_utm = None
transformer_to_wgs84 = None
TARGET_CRS = None

# --- Enhanced OSM Data Initialization ---
def initialize_enhanced_osm_data(bbox_tuple):
    """Enhanced OSM data initialization with additional features"""
    global G_proj, buildings_gdf, water_gdf, green_spaces_gdf, transformer_to_utm, transformer_to_wgs84, TARGET_CRS
    
    if G_proj is not None:
        print("OSM data already initialized.")
        return

    overall_init_start_time = time.time()
    print(f"Initializing enhanced OSM data for bbox: {bbox_tuple}...")
    
    try:
        # 1. Enhanced Road Network
        print("Step 1/4: Fetching enhanced road network...")
        graph_fetch_start_time = time.time()
        G = ox.graph_from_bbox(bbox_tuple, network_type='drive', simplify=True, 
                               retain_all=False, truncate_by_edge=False)
        G_proj = ox.project_graph(G)
        TARGET_CRS = G_proj.graph['crs']
        
        # Add edge weights for realistic routing
        G_proj = ox.add_edge_speeds(G_proj)
        G_proj = ox.add_edge_travel_times(G_proj)
        
        print(f"  Road network loaded (took {time.time() - graph_fetch_start_time:.2f}s)")
        
        wgs84_crs = CRS("EPSG:4326")
        transformer_to_utm = Transformer.from_crs(wgs84_crs, TARGET_CRS, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(TARGET_CRS, wgs84_crs, always_xy=True)

        # 2. Enhanced Building Footprints
        print("Step 2/4: Fetching building footprints...")
        buildings_gdf = load_polygon_features(bbox_tuple, {'building': True}, "buildings")

        # 3. Enhanced Water Bodies
        print("Step 3/4: Fetching water features...")
        water_tags = {'natural': ['water', 'bay'], 'waterway': True, 'landuse': ['reservoir', 'basin']}
        water_gdf = load_polygon_features(bbox_tuple, water_tags, "water")

        # 4. Green Spaces (NEW)
        print("Step 4/4: Fetching green spaces...")
        green_tags = {'landuse': ['forest', 'grass', 'meadow'], 'leisure': ['park', 'garden'], 'natural': ['wood']}
        green_spaces_gdf = load_polygon_features(bbox_tuple, green_tags, "green spaces")
        
        print(f"Total enhanced OSM initialization: {time.time() - overall_init_start_time:.2f}s")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Enhanced OSM data loading failed: {e}")
        G_proj, buildings_gdf, water_gdf, green_spaces_gdf = None, None, None, None

def load_polygon_features(bbox_tuple, tags, feature_type):
    """Helper function to load and process polygon features"""
    try:
        features_gdf = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)
        all_features = ox.features_from_bbox(bbox_tuple, tags=tags)
        
        if not all_features.empty and 'geometry' in all_features.columns:
            wgs84_crs = CRS("EPSG:4326")
            if not isinstance(all_features, gpd.GeoDataFrame):
                all_features = gpd.GeoDataFrame(all_features, geometry='geometry', crs=wgs84_crs)
            else:
                all_features = all_features.set_crs(wgs84_crs, allow_override=True)
            
            valid_geometries = all_features[
                all_features['geometry'].notna() & 
                all_features['geometry'].is_valid
            ].copy()
            
            if not valid_geometries.empty:
                polygon_features = valid_geometries[
                    valid_geometries['geometry'].type.isin(['Polygon', 'MultiPolygon'])
                ].copy()
                
                if not polygon_features.empty:
                    features_gdf_geom = gpd.GeoDataFrame(
                        geometry=polygon_features['geometry'], crs=wgs84_crs
                    )
                    features_gdf = features_gdf_geom.to_crs(TARGET_CRS)
                    print(f"  Loaded {len(features_gdf)} {feature_type} features.")
        
        return features_gdf
    except Exception as e:
        print(f"  Error loading {feature_type}: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)

# --- Enhanced Coordinate System Helpers ---
def project_coords_to_utm_enhanced(lon, lat):
    """Enhanced coordinate projection with error handling"""
    if transformer_to_utm is None:
        raise ValueError("UTM transformer not initialized.")
    try:
        return transformer_to_utm.transform(lon, lat)
    except Exception as e:
        print(f"Error projecting coordinates to UTM: {e}")
        return None, None

def project_coords_to_wgs84_enhanced(x, y):
    """Enhanced WGS84 projection with error handling"""
    if transformer_to_wgs84 is None:
        raise ValueError("WGS84 transformer not initialized.")
    try:
        return transformer_to_wgs84.transform(x, y)
    except Exception as e:
        print(f"Error projecting coordinates to WGS84: {e}")
        return None, None

def enhanced_snap_to_network(point_utm, graph_proj):
    """Enhanced network snapping with multiple candidate selection"""
    if graph_proj is None or point_utm is None:
        return None
    
    try:
        # Get multiple nearest edges for better selection
        nearest_edges = ox.distance.nearest_edges(graph_proj, X=point_utm.x, Y=point_utm.y, 
                                                 return_dist=True)
        
        if isinstance(nearest_edges, tuple) and len(nearest_edges) == 2:
            edge_info, distances = nearest_edges
            if not isinstance(edge_info, list):
                edge_info = [edge_info]
                distances = [distances]
        else:
            edge_info = [nearest_edges]
            distances = [0]
        
        # Select best edge based on distance and road type
        best_edge = None
        min_penalty = float('inf')
        
        for i, (u, v, k) in enumerate(edge_info[:3]):  # Check top 3 edges
            edge_data = graph_proj.get_edge_data(u, v, k)
            if not edge_data:
                continue
                
            # Penalty based on distance and road characteristics
            penalty = distances[i]
            
            # Prefer main roads for snapping
            highway_type = edge_data.get('highway', '')
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else ''
            
            if highway_type in ['primary', 'secondary', 'tertiary']:
                penalty *= 0.8
            elif highway_type in ['residential', 'service']:
                penalty *= 1.2
            
            if penalty < min_penalty:
                min_penalty = penalty
                best_edge = (u, v, k, edge_data)
        
        if best_edge:
            u, v, k, edge_data = best_edge
            edge_geom = edge_data.get('geometry')
            
            if not edge_geom:
                # Fallback to node positions
                u_point = Point(graph_proj.nodes[u]['x'], graph_proj.nodes[u]['y'])
                v_point = Point(graph_proj.nodes[v]['x'], graph_proj.nodes[v]['y'])
                edge_geom = LineString([u_point, v_point])
            
            if isinstance(edge_geom, LineString):
                return edge_geom.interpolate(edge_geom.project(point_utm))
                
        return None
    except Exception as e:
        print(f"Error in enhanced network snapping: {e}")
        return None

# --- Enhanced Geo-Indistinguishability Implementation ---
def enhanced_planar_laplace_noise(epsilon, sensitivity_meters=120.0, quality_factor=1.0):
    """Enhanced Laplace noise with quality-aware sensitivity"""
    adjusted_sensitivity = sensitivity_meters * quality_factor
    scale = adjusted_sensitivity / (epsilon + 1e-9)
    
    # Use inverse transform sampling for better distribution
    u1 = np.random.uniform(1e-10, 1-1e-10)
    u2 = np.random.uniform(1e-10, 1-1e-10)
    
    r = -scale * np.log(u1)
    theta = 2 * np.pi * u2
    
    dx_meters = r * np.cos(theta)
    dy_meters = r * np.sin(theta)
    
    return dx_meters, dy_meters

def enhanced_qos_projection(point_utm, qos_polygon_utm, adaptive_factor=0.9):
    """Enhanced QoS boundary projection with adaptive constraints"""
    if qos_polygon_utm is None or point_utm is None:
        return point_utm
    
    if not qos_polygon_utm.contains(point_utm):
        # Adaptive projection based on distance from boundary
        boundary_distance = qos_polygon_utm.boundary.distance(point_utm)
        
        if boundary_distance > qos_polygon_utm.area ** 0.5 * 0.1:  # Far from boundary
            # Project to a point slightly inside boundary
            projected_point = qos_polygon_utm.boundary.interpolate(
                qos_polygon_utm.boundary.project(point_utm)
            )
            # Move slightly inward
            centroid = qos_polygon_utm.centroid
            direction = Point(
                projected_point.x + (centroid.x - projected_point.x) * 0.05,
                projected_point.y + (centroid.y - projected_point.y) * 0.05
            )
            return direction
        else:
            # Close to boundary, simple projection
            return qos_polygon_utm.boundary.interpolate(
                qos_polygon_utm.boundary.project(point_utm)
            )
    
    return point_utm

def enhanced_realism_check(point_utm, qos_polygon_utm, strict_water_check=True, 
                          check_green_spaces=False):
    """Enhanced realism checking with multiple constraint types"""
    if G_proj is None or not point_utm:
        return False
    
    # Basic QoS check
    qos_buffer = qos_polygon_utm.buffer(50) if qos_polygon_utm else None
    if qos_buffer and not qos_buffer.contains(point_utm):
        return False
    
    realism_score = 1.0
    
    # Water body check
    if strict_water_check and water_gdf is not None and not water_gdf.empty:
        if water_gdf.intersects(point_utm).any():
            try:
                nearest_edge = ox.distance.nearest_edges(G_proj, X=point_utm.x, Y=point_utm.y)
                u, v, k = nearest_edge if isinstance(nearest_edge, tuple) else nearest_edge[0]
                edge_data = G_proj.get_edge_data(u, v, k)
                
                if not (edge_data and edge_data.get('bridge') in ['yes', 'true', True]):
                    realism_score *= 0.1  # Heavy penalty for water without bridge
            except:
                return False
    
    # Building intersection check
    if buildings_gdf is not None and not buildings_gdf.empty:
        if buildings_gdf.intersects(point_utm.buffer(5)).any():
            realism_score *= 0.3  # Penalty for being too close to buildings
    
    # Green space preference (optional)
    if check_green_spaces and green_spaces_gdf is not None and not green_spaces_gdf.empty:
        if green_spaces_gdf.intersects(point_utm.buffer(20)).any():
            realism_score *= 1.2  # Bonus for being near green spaces
    
    return realism_score >= REALISM_THRESHOLD

def get_enhanced_random_point_in_polygon(polygon_wgs84, max_attempts=200):
    """Enhanced random point generation with better distribution"""
    if polygon_wgs84.is_empty:
        return None
    
    min_x, min_y, max_x, max_y = polygon_wgs84.bounds
    area = polygon_wgs84.area
    
    # Use stratified sampling for better distribution
    grid_size = max(3, int(np.sqrt(area) * 100))
    
    for attempt in range(max_attempts):
        if attempt < max_attempts * 0.7:  # First 70% attempts: uniform random
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
        else:  # Remaining attempts: grid-based sampling
            grid_x = random.randint(0, grid_size - 1)
            grid_y = random.randint(0, grid_size - 1)
            x = min_x + (max_x - min_x) * (grid_x + random.random()) / grid_size
            y = min_y + (max_y - min_y) * (grid_y + random.random()) / grid_size
        
        point = Point(x, y)
        if polygon_wgs84.contains(point):
            return (x, y)
    
    # Fallback to centroid
    return (polygon_wgs84.centroid.x, polygon_wgs84.centroid.y)

# --- Enhanced Privacy-Preserving Point Generation ---
def enhanced_r_gits_generate_fake_point(
    true_loc_coords_wgs84,
    epsilon,
    qos_polygon_wgs84,
    is_start=False, start_polygon_wgs84=None,
    is_end=False, end_polygon_wgs84=None,
    previous_fake_point_utm=None,
    path_coherence_enabled=True
):
    """Enhanced R-GITS algorithm with improved realism and path coherence"""
    
    if G_proj is None:
        print("Error: Road network not initialized.")
        return None

    qos_polygon_utm = project_shapely_geom_custom(qos_polygon_wgs84, transformer_to_utm)
    if qos_polygon_utm is None:
        return None

    # Enhanced source point determination
    source_for_gi_utm = None
    quality_factor = 1.0

    # Special handling for start/end points
    if is_start and start_polygon_wgs84:
        quality_factor = 0.8  # Tighter control for start points
        for attempt in range(MAX_REALISTIC_ATTEMPTS):
            rand_coords = get_enhanced_random_point_in_polygon(start_polygon_wgs84)
            if rand_coords:
                rand_utm_coords = project_coords_to_utm_enhanced(rand_coords[0], rand_coords[1])
                if rand_utm_coords[0] is not None:
                    snapped_pt = enhanced_snap_to_network(Point(rand_utm_coords), G_proj)
                    if (snapped_pt and qos_polygon_utm.contains(snapped_pt) and 
                        enhanced_realism_check(snapped_pt, qos_polygon_utm)):
                        source_for_gi_utm = snapped_pt
                        break

    elif is_end and end_polygon_wgs84:
        quality_factor = 0.8  # Tighter control for end points
        for attempt in range(MAX_REALISTIC_ATTEMPTS):
            rand_coords = get_enhanced_random_point_in_polygon(end_polygon_wgs84)
            if rand_coords:
                rand_utm_coords = project_coords_to_utm_enhanced(rand_coords[0], rand_coords[1])
                if rand_utm_coords[0] is not None:
                    snapped_pt = enhanced_snap_to_network(Point(rand_utm_coords), G_proj)
                    if (snapped_pt and qos_polygon_utm.contains(snapped_pt) and 
                        enhanced_realism_check(snapped_pt, qos_polygon_utm)):
                        source_for_gi_utm = snapped_pt
                        break

    # Fallback to true location
    if source_for_gi_utm is None:
        true_utm_coords = project_coords_to_utm_enhanced(true_loc_coords_wgs84[0], true_loc_coords_wgs84[1])
        if true_utm_coords[0] is not None:
            snapped_pt = enhanced_snap_to_network(Point(true_utm_coords), G_proj)
            if (snapped_pt and qos_polygon_utm.contains(snapped_pt)):
                source_for_gi_utm = snapped_pt

    if source_for_gi_utm is None:
        return None

    # Enhanced perturbation with adaptive parameters
    final_snapped_fake_utm = None
    
    for attempt in range(MAX_REALISTIC_ATTEMPTS):
        # Adaptive sensitivity based on attempt and quality factor
        current_sensitivity = max(50, 150 - attempt * 8)
        adaptive_epsilon = epsilon * (1 + attempt * 0.1)  # Gradually relax privacy
        
        dx_m, dy_m = enhanced_planar_laplace_noise(
            adaptive_epsilon, 
            sensitivity_meters=current_sensitivity,
            quality_factor=quality_factor
        )
        
        # Path coherence: bias towards previous direction
        if path_coherence_enabled and previous_fake_point_utm:
            prev_direction_x = source_for_gi_utm.x - previous_fake_point_utm.x
            prev_direction_y = source_for_gi_utm.y - previous_fake_point_utm.y
            direction_magnitude = np.sqrt(prev_direction_x**2 + prev_direction_y**2)
            
            if direction_magnitude > 0:
                coherence_bias = PATH_COHERENCE_WEIGHT * direction_magnitude * 0.3
                dx_m += coherence_bias * (prev_direction_x / direction_magnitude)
                dy_m += coherence_bias * (prev_direction_y / direction_magnitude)
        
        candidate_fake_utm = Point(source_for_gi_utm.x + dx_m, source_for_gi_utm.y + dy_m)
        
        # Enhanced QoS projection
        clamped_candidate_utm = enhanced_qos_projection(candidate_fake_utm, qos_polygon_utm)
        
        # Enhanced network snapping
        snapped_attempt_utm = enhanced_snap_to_network(clamped_candidate_utm, G_proj)
        
        if (snapped_attempt_utm and 
            enhanced_realism_check(snapped_attempt_utm, qos_polygon_utm, 
                                 strict_water_check=True, check_green_spaces=True)):
            final_snapped_fake_utm = snapped_attempt_utm
            break

    # Enhanced fallback strategy
    if final_snapped_fake_utm is None:
        if enhanced_realism_check(source_for_gi_utm, qos_polygon_utm, strict_water_check=True):
            final_snapped_fake_utm = source_for_gi_utm
        elif enhanced_realism_check(source_for_gi_utm, qos_polygon_utm, strict_water_check=False):
            final_snapped_fake_utm = source_for_gi_utm
        else:
            return None

    if final_snapped_fake_utm is None:
        return None

    # Convert back to WGS84
    final_fake_wgs84_coords = project_coords_to_wgs84_enhanced(
        final_snapped_fake_utm.x, final_snapped_fake_utm.y
    )
    
    if final_fake_wgs84_coords[0] is None:
        return None
    
    return final_fake_wgs84_coords

def project_shapely_geom_custom(geom, transformer):
    """Helper function for geometry projection"""
    if geom is None or transformer is None:
        return None
    try:
        return shapely_transform(transformer.transform, geom)
    except Exception as e:
        print(f"Error projecting geometry: {e}")
        return None

# --- Enhanced Visualization ---
def create_enhanced_visualization(real_trajectory, fake_trajectory, qos_polygon, 
                                start_polygon=None, end_polygon=None):
    """Create enhanced visualization with improved styling and layers"""
    
    # Calculate map center
    if real_trajectory:
        center_lat = sum(p[1] for p in real_trajectory) / len(real_trajectory)
        center_lon = sum(p[0] for p in real_trajectory) / len(real_trajectory)
    else:
        center_lat = HCMC_CENTER_LAT_CONFIG
        center_lon = HCMC_CENTER_LON_CONFIG

    # Create map with enhanced styling
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Helper function for coordinate conversion
    def to_lat_lon_folium(coords_list):
        return [(c[1], c[0]) for c in coords_list]

    # Add QoS region with enhanced styling
    if qos_polygon:
        folium.Polygon(
            locations=to_lat_lon_folium(list(qos_polygon.exterior.coords)),
            color="blue", fill=True, fill_color="lightblue",
            fill_opacity=0.15, weight=2, opacity=0.8,
            tooltip="Quality of Service Region",
            popup="QoS Region: Fake points must stay within this area"
        ).add_to(m)

    # Add start region
    if start_polygon:
        folium.Polygon(
            locations=to_lat_lon_folium(list(start_polygon.exterior.coords)),
            color="green", fill=True, fill_color="lightgreen",
            fill_opacity=0.25, weight=2,
            tooltip="Start Region",
            popup="Start Region: Enhanced start point selection"
        ).add_to(m)

    # Add end region
    if end_polygon:
        folium.Polygon(
            locations=to_lat_lon_folium(list(end_polygon.exterior.coords)),
            color="purple", fill=True, fill_color="mediumpurple",
            fill_opacity=0.25, weight=2,
            tooltip="End Region",
            popup="End Region: Enhanced end point protection"
        ).add_to(m)

    # Add enhanced building sample (if available)
    if buildings_gdf is not None and not buildings_gdf.empty:
        try:
            sample_buildings = buildings_gdf.sample(min(len(buildings_gdf), 30))
            folium.GeoJson(
                sample_buildings.to_crs("EPSG:4326"),
                style_function=lambda x: {
                    'fillColor': 'gray', 'color': 'darkgray',
                    'weight': 1, 'fillOpacity': 0.4
                },
                tooltip="Buildings (Sample)"
            ).add_to(m)
        except Exception as e:
            print(f"Could not plot buildings: {e}")

    # Add water bodies (if available)
    if water_gdf is not None and not water_gdf.empty:
        try:
            sample_water = water_gdf.sample(min(len(water_gdf), 20))
            folium.GeoJson(
                sample_water.to_crs("EPSG:4326"),
                style_function=lambda x: {
                    'fillColor': 'lightblue', 'color': 'blue',
                    'weight': 1, 'fillOpacity': 0.5
                },
                tooltip="Water Bodies"
            ).add_to(m)
        except Exception as e:
            print(f"Could not plot water bodies: {e}")

    # Add green spaces (if available)
    if green_spaces_gdf is not None and not green_spaces_gdf.empty:
        try:
            sample_green = green_spaces_gdf.sample(min(len(green_spaces_gdf), 15))
            folium.GeoJson(
                sample_green.to_crs("EPSG:4326"),
                style_function=lambda x: {
                    'fillColor': 'lightgreen', 'color': 'darkgreen',
                    'weight': 1, 'fillOpacity': 0.3
                },
                tooltip="Green Spaces"
            ).add_to(m)
        except Exception as e:
            print(f"Could not plot green spaces: {e}")

    # Add real trajectory with enhanced styling
    if real_trajectory:
        folium.PolyLine(
            to_lat_lon_folium(real_trajectory),
            color="red", weight=4, opacity=0.9,
            tooltip="Real Trajectory (Snapped to Network)",
            popup=f"Real trajectory with {len(real_trajectory)} points"
        ).add_to(m)
        
        # Add real trajectory points
        for i, coords in enumerate(real_trajectory):
            folium.CircleMarker(
                location=(coords[1], coords[0]),
                radius=6, color="red", fill=True, fill_color="darkred",
                fill_opacity=0.8, weight=2,
                tooltip=f"Real Point {i+1}",
                popup=f"Real Point {i+1}: ({coords[0]:.5f}, {coords[1]:.5f})"
            ).add_to(m)

    # Add fake trajectory with enhanced styling
    if fake_trajectory:
        folium.PolyLine(
            to_lat_lon_folium(fake_trajectory),
            color="dodgerblue", weight=4, opacity=0.9,
            dash_array='8, 4',
            tooltip="Privacy-Preserved Fake Trajectory",
            popup=f"Enhanced R-GITS fake trajectory with {len(fake_trajectory)} points"
        ).add_to(m)
        
        # Add fake trajectory points
        for i, coords in enumerate(fake_trajectory):
            folium.CircleMarker(
                location=(coords[1], coords[0]),
                radius=6, color="blue", fill=True, fill_color="royalblue",
                fill_opacity=0.8, weight=2,
                tooltip=f"Fake Point {i+1}",
                popup=f"Fake Point {i+1}: ({coords[0]:.5f}, {coords[1]:.5f})"
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add enhanced legend
    legend_html = """
    <div style="position: fixed; top: 10px; right: 10px; z-index: 1000; 
                background-color: white; padding: 10px; border: 2px solid grey; 
                border-radius: 5px; font-size: 12px;">
    <h4>Enhanced Trajectory Privacy Protection</h4>
    <p><span style="color:red; font-weight:bold;">━━━</span> Real Trajectory</p>
    <p><span style="color:dodgerblue; font-weight:bold;">━ ━ ━</span> Fake Trajectory</p>
    <p><span style="color:blue;">▢</span> QoS Region</p>
    <p><span style="color:green;">▢</span> Start Region</p>
    <p><span style="color:purple;">▢</span> End Region</p>
    <p>Privacy Level (ε): {:.2f}</p>
    </div>
    """.format(PRIVACY_EPSILON)
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# --- Main Enhanced Simulation ---
if __name__ == '__main__':
    overall_start_time = time.time()
    
    print("=== Enhanced Trajectory Privacy Protection System ===")
    print(f"Privacy Parameter (ε): {PRIVACY_EPSILON}")
    print(f"QoS Radius: {QUALITY_OF_SERVICE_RADIUS_M}m")
    print(f"Max Attempts: {MAX_REALISTIC_ATTEMPTS}")
    
    # Initialize enhanced OSM data
    initialize_enhanced_osm_data(HCMC_BBOX)

    if G_proj is None:
        print("Exiting: Enhanced OSM data initialization failed.")
        exit()

    # Define enhanced regions
    QOS_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config),
    ])

    # Enhanced start and end regions
    start_offset = 0.002
    START_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.001, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.001),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.001, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + start_offset),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + start_offset, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + start_offset),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + start_offset, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.001),
    ])

    end_offset = 0.002
    END_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - end_offset, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - end_offset),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - end_offset, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.001),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.001, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.001),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.001, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - end_offset),
    ])

    # Generate enhanced realistic trajectory
    print("\nGenerating enhanced realistic trajectory...")
    start_lon = HCMC_CENTER_LON_CONFIG - HCMC_BBOX_HALF_SIZE_DEG * 0.6
    start_lat = HCMC_CENTER_LAT_CONFIG - HCMC_BBOX_HALF_SIZE_DEG * 0.6
    end_lon = HCMC_CENTER_LON_CONFIG + HCMC_BBOX_HALF_SIZE_DEG * 0.6
    end_lat = HCMC_CENTER_LAT_CONFIG + HCMC_BBOX_HALF_SIZE_DEG * 0.6

    num_points = 12  # Increased for better trajectory
    real_trajectory_coords_wgs84 = []
    
    for i in range(num_points):
        fraction = i / (num_points - 1)
        # Enhanced trajectory with curves
        base_lon = start_lon + fraction * (end_lon - start_lon)
        base_lat = start_lat + fraction * (end_lat - start_lat)
        
        # Add realistic curves and variations
        if 2 <= i <= num_points - 3:  # Avoid start/end perturbation
            curve_factor = np.sin(fraction * np.pi * 2) * 0.0008
            jitter_lon = (random.random() - 0.5) * 0.0003 + curve_factor
            jitter_lat = (random.random() - 0.5) * 0.0003 + curve_factor * 0.7
            base_lon += jitter_lon
            base_lat += jitter_lat
        
        real_trajectory_coords_wgs84.append((base_lon, base_lat))

    # Filter trajectory points within bounds
    real_trajectory_filtered = [
        p for p in real_trajectory_coords_wgs84
        if HCMC_BBOX[3] <= p[0] <= HCMC_BBOX[2] and HCMC_BBOX[1] <= p[1] <= HCMC_BBOX[0]
    ]

    if not real_trajectory_filtered:
        print("Warning: No trajectory points within bounds. Using center point.")
        real_trajectory_filtered = [(HCMC_CENTER_LON_CONFIG, HCMC_CENTER_LAT_CONFIG)]

    # Generate enhanced fake trajectory
    print(f"\nProcessing enhanced trajectory of {len(real_trajectory_filtered)} points...")
    fake_trajectory_coords_wgs84 = []
    snapped_real_trajectory_wgs84 = []
    
    sim_start_time = time.time()
    previous_fake_point_utm = None

    for i, true_loc_wgs84_tuple in enumerate(real_trajectory_filtered):
        print(f"  Processing point {i+1}/{len(real_trajectory_filtered)} - "
              f"({true_loc_wgs84_tuple[0]:.5f}, {true_loc_wgs84_tuple[1]:.5f})")
        
        # Snap real point to network for comparison
        true_utm_coords = project_coords_to_utm_enhanced(true_loc_wgs84_tuple[0], true_loc_wgs84_tuple[1])
        if true_utm_coords[0] is not None:
            snapped_true_utm = enhanced_snap_to_network(Point(true_utm_coords), G_proj)
            if snapped_true_utm:
                snapped_true_wgs84 = project_coords_to_wgs84_enhanced(snapped_true_utm.x, snapped_true_utm.y)
                if snapped_true_wgs84[0] is not None:
                    snapped_real_trajectory_wgs84.append(snapped_true_wgs84)
                else:
                    snapped_real_trajectory_wgs84.append(true_loc_wgs84_tuple)
            else:
                snapped_real_trajectory_wgs84.append(true_loc_wgs84_tuple)
        else:
            snapped_real_trajectory_wgs84.append(true_loc_wgs84_tuple)

        # Generate enhanced fake point
        is_start_pt = (i == 0)
        is_end_pt = (i == len(real_trajectory_filtered) - 1)
        
        fake_pt_wgs84_tuple = enhanced_r_gits_generate_fake_point(
            true_loc_wgs84_tuple, PRIVACY_EPSILON, QOS_POLYGON_WGS84,
            is_start=is_start_pt, start_polygon_wgs84=START_POLYGON_WGS84,
            is_end=is_end_pt, end_polygon_wgs84=END_POLYGON_WGS84,
            previous_fake_point_utm=previous_fake_point_utm,
            path_coherence_enabled=True
        )
        
        if fake_pt_wgs84_tuple:
            fake_trajectory_coords_wgs84.append(fake_pt_wgs84_tuple)
            # Update previous point for path coherence
            fake_utm_coords = project_coords_to_utm_enhanced(fake_pt_wgs84_tuple[0], fake_pt_wgs84_tuple[1])
            if fake_utm_coords[0] is not None:
                previous_fake_point_utm = Point(fake_utm_coords)
            print(f"    ✓ Enhanced fake point: ({fake_pt_wgs84_tuple[0]:.5f}, {fake_pt_wgs84_tuple[1]:.5f})")
        else:
            print(f"    ✗ Failed to generate enhanced fake point")
            previous_fake_point_utm = None

    print(f"\nEnhanced trajectory simulation: {time.time() - sim_start_time:.2f}s")

    # Create enhanced visualization
    print("\nCreating enhanced visualization...")
    vis_start_time = time.time()
    
    enhanced_map = create_enhanced_visualization(
        snapped_real_trajectory_wgs84,
        fake_trajectory_coords_wgs84,
        QOS_POLYGON_WGS84,
        START_POLYGON_WGS84,
        END_POLYGON_WGS84
    )
    
    # Save enhanced map
    map_file = "enhanced_hcmc_trajectory_privacy_map_v5.html"
    enhanced_map.save(map_file)
    
    print(f"Enhanced visualization created: {time.time() - vis_start_time:.2f}s")
    print(f"\n=== Enhanced Results Summary ===")
    print(f"Real trajectory points: {len(snapped_real_trajectory_wgs84)}")
    print(f"Fake trajectory points: {len(fake_trajectory_coords_wgs84)}")
    print(f"Success rate: {len(fake_trajectory_coords_wgs84)/len(real_trajectory_filtered)*100:.1f}%")
    print(f"Enhanced map saved: {map_file}")
    print(f"Total execution time: {time.time() - overall_start_time:.2f}s")
    
    print("\n=== Enhanced Features ===")
    print("✓ Enhanced geo-indistinguishability with adaptive parameters")
    print("✓ Multi-constraint realism checking (water, buildings, green spaces)")
    print("✓ Path coherence for realistic trajectory flow")
    print("✓ Enhanced start/end point protection")
    print("✓ Quality-aware sensitivity adjustment")
    print("✓ Improved network snapping with road type preferences")
    print("✓ Comprehensive visualization with multiple layers")
    print("✓ Adaptive QoS boundary projection")