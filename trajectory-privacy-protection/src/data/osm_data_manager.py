from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import osmnx as ox

class OSMDataManager:
    def __init__(self, bbox):
        self.bbox = bbox
        self.graph = None
        self.buildings = None
        self.water_bodies = None

    def initialize_osm_data(self):
        if self.graph is not None:
            print("OSM data already initialized.")
            return

        print(f"Initializing OSM data for bbox: {self.bbox}...")
        self.graph = ox.graph_from_bbox(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], network_type='drive')
        self.buildings = self.fetch_building_footprints()
        self.water_bodies = self.fetch_water_bodies()

    def fetch_building_footprints(self):
        print("Fetching building footprints...")
        tags = {'building': True}
        buildings_gdf = ox.geometries_from_bbox(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], tags=tags)
        return buildings_gdf

    def fetch_water_bodies(self):
        print("Fetching water bodies...")
        tags = {'natural': ['water', 'bay'], 'waterway': True}
        water_gdf = ox.geometries_from_bbox(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], tags=tags)
        return water_gdf

    def get_nearest_edge(self, point):
        if self.graph is None:
            raise ValueError("OSM data not initialized.")
        nearest_edge = ox.distance.nearest_edges(self.graph, point.x, point.y)
        return nearest_edge

    def snap_to_network(self, point):
        nearest_edge = self.get_nearest_edge(point)
        if nearest_edge:
            u, v, k = nearest_edge
            edge_data = self.graph.get_edge_data(u, v, k)
            edge_geom = edge_data.get('geometry')
            if isinstance(edge_geom, LineString):
                return edge_geom.interpolate(edge_geom.project(point))
        return None

    def get_buildings(self):
        return self.buildings

    def get_water_bodies(self):
        return self.water_bodies

    def get_graph(self):
        return self.graph