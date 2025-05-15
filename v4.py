import numpy as np
import math
import folium # For visualization
from shapely.geometry import Point, Polygon, LineString # For geometric operations
from shapely.ops import transform as shapely_transform
import random # For selecting random points in polygons
import osmnx as ox
import geopandas as gpd
from pyproj import Transformer, CRS
import time

# --- Configuration & Global Variables ---
PRIVACY_EPSILON = 0.5 # Smaller epsilon = more privacy, larger epsilon = less noise
MAX_REALISTIC_ATTEMPTS = 10 # For retrying point generation if initial snap is not ideal
MAX_PLAUSIBLE_SPEED_KMH = 60 # For future path coherence checks

# --- Ho Chi Minh City Coordinates (from user's provided code) ---
HCMC_CENTER_LAT_CONFIG = 10.7769
HCMC_CENTER_LON_CONFIG = 106.7009
offset_qos_deg_config = 0.005 # Approx 500m in degrees

# Define BBOX for OSMnx download based on user's QoS region idea
# Making it slightly larger than the QoS to ensure graph coverage for snapping
HCMC_BBOX = (
    HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config + 0.003,  # North
    HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config - 0.003,  # South
    HCMC_CENTER_LON_CONFIG + offset_qos_deg_config + 0.003,  # East
    HCMC_CENTER_LON_CONFIG - offset_qos_deg_config - 0.003   # West
)

# Global variables for OSM data
G_proj = None # Projected road network graph
buildings_gdf = None # Projected building footprints
water_gdf = None # Projected water bodies
transformer_to_utm = None
transformer_to_wgs84 = None
TARGET_CRS = None # UTM CRS

# --- OSMnx Data Initialization ---
def initialize_osm_data(bbox_tuple):
    global G_proj, buildings_gdf, water_gdf, transformer_to_utm, transformer_to_wgs84, TARGET_CRS
    if G_proj is not None:
        print("OSM data already initialized.")
        return

    overall_init_start_time = time.time()
    print(f"Initializing OSM data for bbox tuple: {bbox_tuple}...")
    print("OSMnx caches downloaded data. Subsequent runs for the SAME bbox will be much faster.")
    try:
        # 1. Road Network
        print("Step 1/3: Fetching and projecting road network graph...")
        graph_fetch_start_time = time.time()
        G = ox.graph_from_bbox(bbox_tuple, network_type='drive', simplify=True, retain_all=False, truncate_by_edge=False)
        G_proj = ox.project_graph(G)
        TARGET_CRS = G_proj.graph['crs'] # Get target CRS from the projected graph
        print(f"  Road network loaded and projected to CRS: {TARGET_CRS} (took {time.time() - graph_fetch_start_time:.2f}s)")
        
        wgs84_crs = CRS("EPSG:4326")
        transformer_to_utm = Transformer.from_crs(wgs84_crs, TARGET_CRS, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(TARGET_CRS, wgs84_crs, always_xy=True)

        # 2. Building Footprints
        print("Step 2/3: Fetching building footprints...")
        features_start_time = time.time()
        tags_building = {'building': True}
        # Initialize as empty GeoDataFrame with the TARGET_CRS
        buildings_gdf = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS) 

        try:
            buildings_all_tags = ox.features_from_bbox(bbox_tuple, tags=tags_building)
            if not buildings_all_tags.empty and 'geometry' in buildings_all_tags.columns:
                # Filter for non-null and valid geometries first
                # Ensure the geometry column is treated as such by GeoPandas
                if not isinstance(buildings_all_tags, gpd.GeoDataFrame):
                    buildings_all_tags = gpd.GeoDataFrame(buildings_all_tags, geometry='geometry', crs=wgs84_crs)
                else:
                    buildings_all_tags = buildings_all_tags.set_crs(wgs84_crs, allow_override=True)

                valid_geometries = buildings_all_tags[buildings_all_tags['geometry'].notna() & buildings_all_tags['geometry'].is_valid].copy()
                
                if not valid_geometries.empty:
                    # Select only Polygon/MultiPolygon types
                    current_buildings_gdf = valid_geometries[valid_geometries['geometry'].type.isin(['Polygon', 'MultiPolygon'])].copy()
                    
                    if not current_buildings_gdf.empty:
                        # The CRS should already be WGS84 from features_from_bbox
                        # current_buildings_gdf = current_buildings_gdf.set_crs(wgs84_crs, allow_override=True) # Already set or inferred
                        buildings_gdf = current_buildings_gdf.to_crs(TARGET_CRS) # Project
                        print(f"  Loaded and projected {len(buildings_gdf)} building features.")
        except Exception as e_feat:
            print(f"  Error during building feature processing: {e_feat}")
            # buildings_gdf remains an empty GeoDataFrame with the correct CRS

        if buildings_gdf.empty: 
            print("  No valid building Polygon/MultiPolygon features found or processed.")
        print(f"  Building features step took {time.time() - features_start_time:.2f}s")

        # 3. Water Bodies
        print("Step 3/3: Fetching water features...")
        water_features_start_time = time.time()
        tags_water = {'natural': ['water', 'bay'], 'waterway': True, 'landuse': ['reservoir', 'basin']}
        # Initialize as empty GeoDataFrame with the TARGET_CRS
        water_gdf = gpd.GeoDataFrame(geometry=[], crs=TARGET_CRS)

        try:
            water_all_tags = ox.features_from_bbox(bbox_tuple, tags=tags_water)
            if not water_all_tags.empty and 'geometry' in water_all_tags.columns:
                if not isinstance(water_all_tags, gpd.GeoDataFrame):
                    water_all_tags = gpd.GeoDataFrame(water_all_tags, geometry='geometry', crs=wgs84_crs)
                else:
                    water_all_tags = water_all_tags.set_crs(wgs84_crs, allow_override=True)
                
                valid_geometries_water = water_all_tags[water_all_tags['geometry'].notna() & water_all_tags['geometry'].is_valid].copy()

                if not valid_geometries_water.empty:
                    current_water_gdf = valid_geometries_water[valid_geometries_water['geometry'].type.isin(['Polygon', 'MultiPolygon'])].copy()

                    if not current_water_gdf.empty:
                        # current_water_gdf = current_water_gdf.set_crs(wgs84_crs, allow_override=True) # Already set or inferred
                        water_gdf = current_water_gdf.to_crs(TARGET_CRS)
                        print(f"  Loaded and projected {len(water_gdf)} water features.")
        except Exception as e_feat_water:
            print(f"  Error during water feature processing: {e_feat_water}")

        if water_gdf.empty: 
            print("  No valid water Polygon/MultiPolygon features found or processed.")
        print(f"  Water features step took {time.time() - water_features_start_time:.2f}s")
        
        print(f"Total OSM data initialization time: {time.time() - overall_init_start_time:.2f} seconds.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load OSM data (Overall): {e}")
        G_proj, buildings_gdf, water_gdf = None, None, None 

# --- Coordinate System and Snapping Helpers ---
def project_coords_to_utm_custom(lon, lat):
    if transformer_to_utm is None: raise ValueError("transformer_to_utm not initialized.")
    return transformer_to_utm.transform(lon, lat)

def project_coords_to_wgs84_custom(x, y):
    if transformer_to_wgs84 is None: raise ValueError("transformer_to_wgs84 not initialized.")
    return transformer_to_wgs84.transform(x, y)

def project_shapely_geom_custom(geom, transformer):
    if geom is None or transformer is None: return None
    return shapely_transform(transformer.transform, geom)

def snap_point_to_network_utm(point_utm, graph_proj):
    if graph_proj is None or point_utm is None: return None
    try:
        edge_uvk = ox.distance.nearest_edges(graph_proj, X=point_utm.x, Y=point_utm.y)
        if not edge_uvk: return None
        
        u, v, k = edge_uvk if isinstance(edge_uvk, tuple) else edge_uvk[0]
        edge_data = graph_proj.get_edge_data(u, v, k)
        edge_geom = edge_data.get('geometry') if edge_data else None

        if not edge_geom: 
            all_edges = graph_proj.edges(keys=True, data=True)
            edge_geom = next((data['geometry'] for u_n, v_n, key_n, data in all_edges if u_n == u and v_n == v and key_n == k and 'geometry' in data), None)
            if not edge_geom: return None
            
        if isinstance(edge_geom, LineString):
            return edge_geom.interpolate(edge_geom.project(point_utm))
        return None
    except Exception: return None

# --- Geo-Indistinguishability and Point Generation ---
def planar_laplace_noise_meters(epsilon, sensitivity_meters=100.0):
    scale = sensitivity_meters / (epsilon + 1e-9)
    u1 = np.random.uniform(-0.5, 0.5)
    u2 = np.random.uniform(-0.5, 0.5)
    dx_meters = scale * np.sign(u1) * np.log(1 - 2 * np.abs(u1))
    dy_meters = scale * np.sign(u2) * np.log(1 - 2 * np.abs(u2))
    return dx_meters, dy_meters

def project_to_qos_boundary_utm(point_utm, qos_polygon_utm):
    if qos_polygon_utm is None or point_utm is None: return point_utm
    if not qos_polygon_utm.contains(point_utm):
        projected_point = qos_polygon_utm.boundary.interpolate(qos_polygon_utm.boundary.project(point_utm))
        return projected_point
    return point_utm

def get_random_point_in_polygon_wgs84(polygon_wgs84): # Expects Shapely Polygon
    min_x, min_y, max_x, max_y = polygon_wgs84.bounds
    attempts = 0
    while attempts < 100: 
        random_point_geom = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon_wgs84.contains(random_point_geom):
            return (random_point_geom.x, random_point_geom.y) # lon, lat tuple
        attempts +=1
    return (polygon_wgs84.centroid.x, polygon_wgs84.centroid.y)


def is_location_realistic_on_network(point_utm, qos_polygon_utm):
    """Checks if a point (already snapped to network, in UTM) is realistic."""
    if G_proj is None or not point_utm: return False 
    if qos_polygon_utm and not qos_polygon_utm.contains(point_utm): return False 

    if water_gdf is not None and not water_gdf.empty and water_gdf.intersects(point_utm).any():
        try:
            nearest_edge_uvk = ox.distance.nearest_edges(G_proj, X=point_utm.x, Y=point_utm.y)
            u,v,k = nearest_edge_uvk if isinstance(nearest_edge_uvk, tuple) else nearest_edge_uvk[0]
            edge_data = G_proj.get_edge_data(u,v,k)
            if not (edge_data and edge_data.get('bridge') in ['yes', 'true', True, 1, '1', 'viaduct']):
                return False 
        except Exception: 
            return False 

    return True

def r_gits_generate_fake_point_on_network(
        true_loc_coords_wgs84, # (lon, lat) tuple
        epsilon,
        qos_polygon_wgs84, # Shapely Polygon in WGS84
        is_start=False, start_polygon_wgs84=None, # Shapely Polygon
        is_end=False, end_polygon_wgs84=None,   # Shapely Polygon
        last_fake_snapped_utm=None): # For future routing

    if G_proj is None:
        print("Error: Road network (G_proj) not initialized for point generation.")
        return None

    qos_polygon_utm = project_shapely_geom_custom(qos_polygon_wgs84, transformer_to_utm)
    source_for_gi_utm = None 

    region_poly_wgs84 = None
    if is_start and start_polygon_wgs84: region_poly_wgs84 = start_polygon_wgs84
    elif is_end and end_polygon_wgs84: region_poly_wgs84 = end_polygon_wgs84

    if region_poly_wgs84:
        for _ in range(MAX_REALISTIC_ATTEMPTS): 
            rand_region_pt_wgs84_coords = get_random_point_in_polygon_wgs84(region_poly_wgs84)
            rand_region_pt_utm_coords = project_coords_to_utm_custom(rand_region_pt_wgs84_coords[0], rand_region_pt_wgs84_coords[1])
            snapped_pt = snap_point_to_network_utm(Point(rand_region_pt_utm_coords), G_proj)
            if snapped_pt and is_location_realistic_on_network(snapped_pt, qos_polygon_utm):
                source_for_gi_utm = snapped_pt
                break
    else: 
        true_loc_utm_coords = project_coords_to_utm_custom(true_loc_coords_wgs84[0], true_loc_coords_wgs84[1])
        source_for_gi_utm = snap_point_to_network_utm(Point(true_loc_utm_coords), G_proj)

    if source_for_gi_utm is None: 
        # Fallback: if snapping fails, use the original projected point (might be off-network)
        source_for_gi_utm = Point(project_coords_to_utm_custom(true_loc_coords_wgs84[0], true_loc_coords_wgs84[1]))

    final_snapped_fake_utm = None
    for attempt in range(MAX_REALISTIC_ATTEMPTS):
        current_sensitivity = 150.0 - attempt * 10 
        dx_m, dy_m = planar_laplace_noise_meters(epsilon, sensitivity_meters=max(20, current_sensitivity))
        candidate_fake_utm = Point(source_for_gi_utm.x + dx_m, source_for_gi_utm.y + dy_m)
        clamped_candidate_fake_utm = project_to_qos_boundary_utm(candidate_fake_utm, qos_polygon_utm)
        if clamped_candidate_fake_utm is None: clamped_candidate_fake_utm = candidate_fake_utm # Should not be None if qos_polygon_utm is valid
        
        snapped_attempt_utm = snap_point_to_network_utm(clamped_candidate_fake_utm, G_proj)
        
        if snapped_attempt_utm and is_location_realistic_on_network(snapped_attempt_utm, qos_polygon_utm):
            final_snapped_fake_utm = snapped_attempt_utm
            break 
    
    if final_snapped_fake_utm is None: # If all attempts to find a perturbed realistic point fail
        # Fallback to the (snapped) source point for Geo-I if it's realistic
        if source_for_gi_utm and is_location_realistic_on_network(source_for_gi_utm, qos_polygon_utm):
            final_snapped_fake_utm = source_for_gi_utm
        else: # Absolute fallback if even the source point isn't good (should be rare)
            return None 

    final_fake_wgs84_coords = project_coords_to_wgs84_custom(final_snapped_fake_utm.x, final_snapped_fake_utm.y)
    return final_fake_wgs84_coords

# --- Simulation & Visualization ---
if __name__ == '__main__':
    overall_start_time = time.time()
    initialize_osm_data(HCMC_BBOX)

    if G_proj is None:
        print("Exiting: OSM data initialization failed. Cannot proceed with simulation.")
        exit()

    QOS_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config),
    ])

    offset_start_end_deg_config = 0.001 
    START_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.0005, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.0005),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.0005, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.0015),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.0015, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.0015),
        (HCMC_CENTER_LON_CONFIG - offset_qos_deg_config + 0.0015, HCMC_CENTER_LAT_CONFIG - offset_qos_deg_config + 0.0005),
    ])

    END_POLYGON_WGS84 = Polygon([
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.0015, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.0015),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.0015, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.0005),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.0005, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.0005),
        (HCMC_CENTER_LON_CONFIG + offset_qos_deg_config - 0.0005, HCMC_CENTER_LAT_CONFIG + offset_qos_deg_config - 0.0015),
    ])
    
    real_trajectory_coords_wgs84 = [
        (HCMC_CENTER_LON_CONFIG - 0.004, HCMC_CENTER_LAT_CONFIG - 0.004),
        (HCMC_CENTER_LON_CONFIG - 0.002, HCMC_CENTER_LAT_CONFIG - 0.001),
        (HCMC_CENTER_LON_CONFIG + 0.001, HCMC_CENTER_LAT_CONFIG + 0.002),
        (HCMC_CENTER_LON_CONFIG + 0.003, HCMC_CENTER_LAT_CONFIG + 0.004) 
    ]
    real_trajectory_coords_wgs84_filtered = [
        p for p in real_trajectory_coords_wgs84
        if HCMC_BBOX[3] <= p[0] <= HCMC_BBOX[2] and HCMC_BBOX[1] <= p[1] <= HCMC_BBOX[0]
    ]
    if not real_trajectory_coords_wgs84_filtered:
         print("Warning: All real trajectory points are outside the OSM data bounding box. Using HCMC center as a single point.")
         real_trajectory_coords_wgs84_filtered = [(HCMC_CENTER_LON_CONFIG, HCMC_CENTER_LAT_CONFIG)]

    fake_trajectory_coords_wgs84 = []
    snapped_real_trajectory_wgs84 = [] 
    
    print(f"\nProcessing trajectory of {len(real_trajectory_coords_wgs84_filtered)} points...")
    sim_start_time = time.time()
    total_points = len(real_trajectory_coords_wgs84_filtered)
    last_fake_snapped_point_utm = None

    for i, true_loc_wgs84_tuple in enumerate(real_trajectory_coords_wgs84_filtered):
        print(f"  Processing point {i+1} of {total_points} (True WGS84: {true_loc_wgs84_tuple[0]:.5f}, {true_loc_wgs84_tuple[1]:.5f})...")
        
        true_loc_utm_pt = Point(project_coords_to_utm_custom(true_loc_wgs84_tuple[0], true_loc_wgs84_tuple[1]))
        snapped_true_utm_pt = snap_point_to_network_utm(true_loc_utm_pt, G_proj)
        if snapped_true_utm_pt:
            snapped_true_wgs84_coords = project_coords_to_wgs84_custom(snapped_true_utm_pt.x, snapped_true_utm_pt.y)
            snapped_real_trajectory_wgs84.append(snapped_true_wgs84_coords)
        else: 
            snapped_real_trajectory_wgs84.append(true_loc_wgs84_tuple)

        is_start_pt = (i == 0)
        is_end_pt = (i == total_points - 1)
        
        fake_pt_wgs84_tuple = r_gits_generate_fake_point_on_network(
            true_loc_wgs84_tuple, PRIVACY_EPSILON, QOS_POLYGON_WGS84,
            is_start=is_start_pt, start_polygon_wgs84=START_POLYGON_WGS84,
            is_end=is_end_pt, end_polygon_wgs84=END_POLYGON_WGS84,
            last_fake_snapped_utm=last_fake_snapped_point_utm)
        
        if fake_pt_wgs84_tuple:
            fake_trajectory_coords_wgs84.append(fake_pt_wgs84_tuple)
            fake_pt_utm_coords = project_coords_to_utm_custom(fake_pt_wgs84_tuple[0], fake_pt_wgs84_tuple[1])
            last_fake_snapped_point_utm = Point(fake_pt_utm_coords) 
            print(f"    -> Fake generated (WGS84): ({fake_pt_wgs84_tuple[0]:.5f}, {fake_pt_wgs84_tuple[1]:.5f})")
        else:
            print(f"    -> Failed to generate valid fake point.")
            last_fake_snapped_point_utm = None

    print(f"Trajectory simulation took {time.time() - sim_start_time:.2f}s")

    map_creation_start_time = time.time()
    map_display_center_lat = (HCMC_BBOX[0] + HCMC_BBOX[1]) / 2
    map_display_center_lon = (HCMC_BBOX[2] + HCMC_BBOX[3]) / 2
    m = folium.Map(location=[map_display_center_lat, map_display_center_lon], zoom_start=15, tiles="OpenStreetMap")

    def to_lat_lon_folium(coords_list_lon_lat): return [(c[1], c[0]) for c in coords_list_lon_lat]

    if QOS_POLYGON_WGS84: folium.Polygon(locations=to_lat_lon_folium(list(QOS_POLYGON_WGS84.exterior.coords)), color="blue", fill=True, fill_color="blue", fill_opacity=0.1, tooltip="QoS Region", name="QoS Region").add_to(m)
    if START_POLYGON_WGS84: folium.Polygon(locations=to_lat_lon_folium(list(START_POLYGON_WGS84.exterior.coords)), color="green", fill=True, fill_color="green", fill_opacity=0.2, tooltip="Start Region", name="Start Region").add_to(m)
    if END_POLYGON_WGS84: folium.Polygon(locations=to_lat_lon_folium(list(END_POLYGON_WGS84.exterior.coords)), color="purple", fill=True, fill_color="purple", fill_opacity=0.2, tooltip="End Region", name="End Region").add_to(m)
    
    if buildings_gdf is not None and not buildings_gdf.empty:
        try: 
            buildings_to_plot = buildings_gdf.sample(min(len(buildings_gdf), 30)) 
            folium.GeoJson(buildings_to_plot.to_crs("EPSG:4326"), style_function=lambda x: {'fillColor': 'grey', 'color':'darkgrey', 'weight': 0.5, 'fillOpacity':0.3}, name="Buildings (Sample from OSMnx)").add_to(m)
        except Exception as e: print(f"Could not plot buildings: {e}")
    if water_gdf is not None and not water_gdf.empty:
        try: 
            water_to_plot = water_gdf.sample(min(len(water_gdf), 30)) 
            folium.GeoJson(water_to_plot.to_crs("EPSG:4326"), style_function=lambda x: {'fillColor': 'lightblue', 'color':'blue', 'weight': 0.5, 'fillOpacity':0.4}, name="Water Bodies (Sample from OSMnx)").add_to(m)
        except Exception as e: print(f"Could not plot water: {e}")

    if snapped_real_trajectory_wgs84:
        folium.PolyLine(to_lat_lon_folium(snapped_real_trajectory_wgs84), color="red", weight=3, opacity=0.8, tooltip="Snapped Real Trajectory", name="Snapped Real Trajectory").add_to(m)
        for i, p_coords in enumerate(snapped_real_trajectory_wgs84): 
            folium.CircleMarker(location=(p_coords[1],p_coords[0]), radius=5, color="red", fill=True,fill_color="darkred",tooltip=f"Real (Snapped) Pt {i+1}").add_to(m)
    
    if fake_trajectory_coords_wgs84:
        folium.PolyLine(to_lat_lon_folium(fake_trajectory_coords_wgs84), color="dodgerblue", weight=3, opacity=0.8, dash_array='10, 5', tooltip="Fake Trajectory (on network)", name="Fake Trajectory").add_to(m)
        for i, p_coords in enumerate(fake_trajectory_coords_wgs84): 
            folium.CircleMarker(location=(p_coords[1],p_coords[0]), radius=5, color="blue", fill=True,fill_color="royalblue",tooltip=f"Fake Pt {i+1}").add_to(m)

    folium.LayerControl().add_to(m) 
    map_file = "hcmc_trajectory_privacy_map_osmnx_realism.html" 
    m.save(map_file)
    print(f"Map creation took {time.time() - map_creation_start_time:.2f}s")
    print(f"\nMap saved to {map_file}")
    print("--- Note on Realism & Traffic Rules ---")
    print("This version uses OSMnx data for road network, buildings, and water bodies.")
    print("Fake points are snapped to roads and checked against water (unless on bridges).")
    print("Full traffic rule adherence (one-ways, etc.) would require route generation between points.")
    print(f"Total script execution time: {time.time() - overall_start_time:.2f} seconds")

