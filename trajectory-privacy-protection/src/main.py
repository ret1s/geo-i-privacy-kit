# filepath: /trajectory-privacy-protection/trajectory-privacy-protection/src/main.py

from core.privacy_engine import PrivacyEngine
from data.osm_data_manager import OSMDataManager
from visualization.map_generator import MapGenerator

def main():
    # Initialize the OSM data manager
    osm_data_manager = OSMDataManager()
    osm_data_manager.load_data()

    # Initialize the privacy engine
    privacy_engine = PrivacyEngine(osm_data_manager)

    # Generate fake trajectories
    fake_trajectories = privacy_engine.generate_fake_trajectories()

    # Visualize the results
    map_generator = MapGenerator()
    map_generator.create_map(fake_trajectories)

if __name__ == "__main__":
    main()