import geopandas as gpd

# import polars as pl


def load_plgns(path):
    if path.suffix == '.gz':
        path = path.as_uri().replace('file', 'gzip')

    gdf = gpd.read_file(path)

    return gdf