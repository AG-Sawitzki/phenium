from functools import partial
from multiprocessing import Pool
from pathlib import Path

import tqdm
from phenium import analysis, io

gdf = io.load_plgns(
    Path(
        "/data/cephfs-1/scratch/projects/sawitzki-phenium/cell-polygon-layers.geojson.gz"
    )
)
gdf = gdf.sort_values(by=["geometry"])
gdf = gdf.reset_index()
func = partial(analysis.foo, gdf=gdf, midx=len(gdf))
res = {}
with Pool(2) as pool:
    for r in tqdm.tqdm(pool.imap(func, gdf.iterrows()), total=len(gdf)):
        res.update(r)
    pool.close()
    pool.join()
