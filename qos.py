import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, LineString
import contextily as ctx
import random
from datetime import datetime
import os
import numpy as np

def visualize_qos_region(location=(10.772, 106.698), radius=50, network_distance=1500):
    """
    Choose a random point on the road network and visualize the QoS region around it.
    
    Parameters:
        location: (lat, lon) tuple for the center of the area to analyze
        radius: Radius in meters for the QoS circle (default: 50m)
        network_distance: Distance in meters to define the road network area
    
    Returns:
        selected_point: The randomly selected point on the road
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs("result", exist_ok=True)
        
        # Download the road network (reduced distance for faster download)
        print(f"Downloading road network for location {location}...")
        G = ox.graph_from_point(location, dist=network_distance, network_type='drive', simplify=True)
        
        if len(G.nodes) == 0:
            print("Error: No nodes found in the road network. Try a different location or larger network_distance.")
            return None
            
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
        
        # Select a random node
        print(f"Selecting random point from {len(G.nodes)} nodes...")
        node_ids = list(G.nodes)
        random_node_id = random.choice(node_ids)
        
        # Make sure the node exists in the GeoDataFrame
        if random_node_id not in nodes_gdf.index:
            print(f"Node {random_node_id} not found in GeoDataFrame. Selecting a different node.")
            for node_id in node_ids:
                if node_id in nodes_gdf.index:
                    random_node_id = node_id
                    break
            else:
                print("Error: No valid nodes found in the GeoDataFrame.")
                return None
                
        random_node = nodes_gdf.loc[random_node_id]
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create point geometry for the selected location
        point = Point(random_node.geometry.x, random_node.geometry.y)
        # Convert to Web Mercator projection for consistent distance calculations
        point_gdf = gpd.GeoDataFrame(geometry=[point], crs=nodes_gdf.crs).to_crs(epsg=3857)
        
        # Calculate center coordinates
        center_x = point_gdf.geometry.x.values[0]
        center_y = point_gdf.geometry.y.values[0]
        
        # Use a smaller padding to make the circle more prominent
        padding = radius * 4
        
        # Plot the entire road network first (safer than clipping)
        print("Plotting road network...")
        edges_gdf.to_crs(epsg=3857).plot(ax=ax, color='gray', linewidth=1.0, alpha=0.5)
        
        # Create buffer (circle) with the specified radius
        print(f"Creating {radius}m buffer around selected point...")
        buffer = point_gdf.buffer(radius)
        buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=point_gdf.crs)
        
        # Plot the buffer circle
        buffer_gdf.plot(ax=ax, color='blue', alpha=0.3, zorder=2)
        buffer_gdf.boundary.plot(ax=ax, color='blue', linewidth=2.5, alpha=0.8, zorder=3)
        
        # Plot the center point
        point_gdf.plot(ax=ax, color='red', markersize=120, marker='*', 
                      edgecolor='black', linewidth=1.5, zorder=4)
        
        # Add a label for the radius
        ax.text(
            center_x, 
            center_y - radius/2,
            f"50m radius", 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            zorder=5
        )
        
        # Add a scale indicator for 50m
        scale_line_start_x = center_x - radius
        scale_line_end_x = center_x
        scale_line_y = center_y - radius * 1.5
        
        # Create and plot scale line
        scale_line = LineString([(scale_line_start_x, scale_line_y), (scale_line_end_x, scale_line_y)])
        scale_line_gdf = gpd.GeoDataFrame(geometry=[scale_line], crs=buffer_gdf.crs)
        scale_line_gdf.plot(ax=ax, color='black', linewidth=3, zorder=5)
        
        # Add scale label
        ax.text(
            scale_line_start_x + (scale_line_end_x - scale_line_start_x)/2, 
            scale_line_y - radius/4,
            "50 meters", 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),
            zorder=5
        )
        
        # Set bounds to the calculated padding - IMPORTANT: do this BEFORE adding basemap
        ax.set_xlim(center_x - padding, center_x + padding)
        ax.set_ylim(center_y - padding, center_y + padding)
        
        # Use a fixed aspect ratio instead of letting GeoPandas calculate it
        ax.set_aspect('equal')
        
        # Higher zoom level for closer view (17 is much closer than 13)
        zoom_level = 17
            
        print(f"Adding basemap with zoom level {zoom_level}...")
        
        # Add basemap
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=zoom_level)
        except Exception as e:
            print(f"Error with CartoDB basemap: {e}")
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom_level)
            except Exception as e:
                print(f"Error with OpenStreetMap basemap: {e}")
                print("Continuing without basemap...")
        
        # Set title and turn off axis
        ax.set_title(f"Quality of Service Region (50m radius)", fontsize=16)
        ax.axis('off')
        
        # Save figure
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"result/{current_time}_qos_region_50m.png"
        print(f"Saving figure to {output_path}...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        print(f"Generated QoS region visualization with 50m radius")
        
        # Display the figure
        plt.show()
        
        return point
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Only visualize with a 50m radius
    visualize_qos_region(radius=150, network_distance=3000)
