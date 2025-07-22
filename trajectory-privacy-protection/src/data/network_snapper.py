from shapely.geometry import Point
import osmnx as ox

class NetworkSnapper:
    def __init__(self, graph):
        self.graph = graph

    def snap_to_network(self, point):
        if not isinstance(point, Point):
            raise ValueError("Input must be a Shapely Point.")
        
        nearest_edge = ox.distance.nearest_edges(self.graph, point.x, point.y)
        if nearest_edge:
            u, v, k = nearest_edge
            edge_data = self.graph.get_edge_data(u, v, k)
            edge_geom = edge_data.get('geometry')
            if edge_geom:
                snapped_point = edge_geom.interpolate(edge_geom.project(point))
                return snapped_point
        return None

    def snap_multiple_points(self, points):
        snapped_points = []
        for point in points:
            snapped_point = self.snap_to_network(point)
            if snapped_point:
                snapped_points.append(snapped_point)
        return snapped_points