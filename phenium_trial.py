from pathlib import Path

import numpy as np
import polars as pl
import scanpy as sc
import anndata as ad
import rapids_singlecell as rsc
import geopandas as gpd
from shapely.geometry import Point
from scipy.io import mmread
from cupyx.scipy.sparse import csr_matrix
# import spatialdata

from phenium.main import Phenium
obj = Phenium(
    SRT_dir='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260716__161300__20260716_ITCAIM_lung-spleen_run1/output-XETG00410__0064543__lung__20260716__161319',
    IHC_path='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260723_ITCAIM_Phenium_Slide1/Scan1/20260723_ITCAIM_Phenium_Slide1_Scan1.er.qptiff',
    output_dir='/data/cephfs-1/scratch/projects/sawitzki-phenium',
    config_path='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/phenium/config.toml',
)

obj.prepare(4,6, quick=True, verbose=True)
obj._run_proseg()

ply_path = obj.output_dir / 'cell-polygon-layers.geojson.gz'

output_dir = Path('/data/cephfs-1/scratch/projects/sawitzki-phenium')

counts = mmread(output_dir / 'counts.mtx.gz').tocsr()
genes = pl.read_parquet(output_dir / 'gene-metadata.parquet')
cells = pl.read_parquet(output_dir / 'cell-metadata.parquet')

gdf = gpd.read_file(
    ply_path.as_uri().replace('file', 'gzip')
)

expr = ((pl.col('cell_id')).is_in(cells['original_cell_id']))
transcripts_in_cells = obj.transcripts.filter(expr)
points = transcripts_in_cells[['x_location', 'y_location']].map_rows(
    lambda r: Point(r[0], r[1])
)
points_gdf = gpd.GeoDataFrame(geometry=points)
for layer in range(max(gdf['layer'])+1):
    check_layer = gdf['layer'] == layer
    check_loc = points_gdf['geometry'].within(gdf[check_layer])
    ...

obs = cells[['original_cell_id', 'cell']]
obs_names = cells['original_cell_id']
obsm = dict(
    spatial = cells[['centroid_x', 'centroid_y', 'centroid_z']].to_numpy()
)

var = genes[['gene', 'total_count', 'expected_assigned_count']]
var_names = genes['gene']
X = csr_matrix(counts)

adata = ad.AnnData(X=X, obs=obs.to_pandas(), var=var.to_pandas(), obsm=obsm)

adata.obs_names = obs_names.to_pandas()
adata.var_names = var_names.to_pandas()

gdata = rsc.get.anndata_to_GPU(adata, convert_all=True, copy=True)

rsc.pp.calculate_qc_metrics(gdata)

rsc.pp.log1p(gdata)

rsc.pp.neighbors(gdata, use_rep='X')
rsc.tl.leiden(gdata)
rsc.tl.umap(gdata)

cdata = rsc.get.anndata_to_CPU(gdata, convert_all=True, copy=True)
sc.pl.umap(cdata, color='leiden', show=False).figure.savefig(output_dir / 'leiden-umap_gpu.png')
