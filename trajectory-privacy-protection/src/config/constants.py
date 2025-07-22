# filepath: /trajectory-privacy-protection/trajectory-privacy-protection/src/config/constants.py

# Constants for trajectory privacy protection project

# Privacy parameters
PRIVACY_EPSILON = 0.5  # Smaller epsilon = more privacy, larger epsilon = less noise
MAX_REALISTIC_ATTEMPTS = 10  # Maximum attempts for generating realistic points

# Quality of Service parameters
QoS_RADIUS_METERS = 800  # Radius for quality of service checks

# Visualization parameters
MAP_ZOOM_LEVEL = 14  # Default zoom level for the map
REAL_TRAJECTORY_COLOR = 'red'  # Color for real trajectories
FAKE_TRAJECTORY_COLOR = 'blue'  # Color for fake trajectories
START_POINT_COLOR = 'green'  # Color for start points
END_POINT_COLOR = 'purple'  # Color for end points

# Geographic constants
HCMC_CENTER_LAT = 10.7769  # Latitude for Ho Chi Minh City center
HCMC_CENTER_LON = 106.7009  # Longitude for Ho Chi Minh City center
OFFSET_QOS_DEG = 0.008  # Offset for QoS region in degrees

# File paths
OUTPUT_MAP_FILE = "output/hcmc_trajectory_privacy_map.html"  # Output file for the generated map
SAMPLE_TRAJECTORIES_FILE = "data/sample_trajectories.json"  # Sample trajectories data file