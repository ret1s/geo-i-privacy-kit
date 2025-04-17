import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# Planar Laplace noise
def planar_laplace_noise(epsilon):
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.exponential(1 / epsilon)
    return r * np.cos(theta), r * np.sin(theta)

# Generate noisy locations
def generate_noisy_locations(real_location, epsilon, scale_factor=1.0, num_samples=200):
    noisy_points = []
    for _ in range(num_samples):
        dx, dy = planar_laplace_noise(epsilon)
        dx *= scale_factor
        dy *= scale_factor
        noisy_lat = real_location[0] + (dy / 111320)
        noisy_lon = real_location[1] + (dx / (40075000 * np.cos(np.radians(real_location[0])) / 360))
        noisy_points.append(Point(noisy_lon, noisy_lat))
    return noisy_points

# Real location in Ho Chi Minh City
real_location = (10.772, 106.698)
real_point = Point(real_location[1], real_location[0])

epsilon = 0.1
scale_factor = 10.0
noisy_points = generate_noisy_locations(real_location, epsilon, scale_factor)

# Create GeoDataFrames
real_gdf = gpd.GeoDataFrame(geometry=[real_point], crs="EPSG:4326")
noisy_gdf = gpd.GeoDataFrame(geometry=noisy_points, crs="EPSG:4326")

# Convert to Web Mercator for map overlay
real_gdf = real_gdf.to_crs(epsg=3857)
noisy_gdf = noisy_gdf.to_crs(epsg=3857)

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the data
noisy_gdf.plot(ax=ax, alpha=0.5, color='blue', markersize=10, label='Noisy Locations')
real_gdf.plot(ax=ax, color='red', markersize=50, label='Real Location')

# Set appropriate bounds before adding basemap
minx, miny, maxx, maxy = noisy_gdf.total_bounds
ax.set_xlim(minx - 1000, maxx + 1000)  # Add some padding
ax.set_ylim(miny - 1000, maxy + 1000)

# Try with a specific zoom level
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=17)

ax.set_title(f"Geo-Indistinguishability in Ho Chi Minh City (ε = {epsilon})")
ax.legend()
plt.axis('off')
plt.tight_layout()
plt.show()