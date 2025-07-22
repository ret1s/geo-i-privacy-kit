from pyproj import Transformer

class CoordinateTransformer:
    def __init__(self, source_crs, target_crs):
        self.transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def transform(self, lon, lat):
        return self.transformer.transform(lon, lat)

    def inverse_transform(self, x, y):
        return self.transformer.transform(x, y, direction='INVERSE')