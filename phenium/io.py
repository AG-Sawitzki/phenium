import pickle
from pathlib import Path, PurePath

import anndata as ad
import geopandas as gpd
import polars as pl
import scanpy as sc
import squidpy as sq


def save_settings(obj, file):
    if obj.machine:
        with open(file, "a") as f:
            f.write(f"\n IST-machine: {obj.machine}")
    if obj.prior_type:
        with open(file, 'a') as f:
            f.write(f'\n prior-type: {obj.prior_type}')
    if obj.exclude:
        with open(file, "a") as f:
            f.write("\n excluded-markers:\n")
            f.writelines(f"{p}\n" for p in obj.exclude)
    if obj.channels:
        with open(file, "a") as f:
            f.write('\n segmentation-channels:\n')
            f.writelines(f"{c}\n" for c in obj.channels)
    if obj.cell_size:
        with open(file, 'a') as f:
            f.write(f'\n cell-size: {obj.cell_size} \n')
    if obj.seed:
        with open(file, 'a') as f:
            f.write(f'\n seed: {obj.seed} \n')


def save_phenium(obj, file):
    with open(file, 'wb') as outp:
        pickle.dump(obj, outp, pickle.DEFAULT_PROTOCOL)


def load_phenium(file):
    with open(file, 'rb') as inp:
        ana = pickle.load(inp)
    return ana


def load_plgns(path):
    assert isinstance(path, PurePath | str), "path must be str or PurePath!"
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == '.gz':
        path = path.as_uri().replace('file', 'gzip')

    gdf = gpd.read_file(path)

    return gdf


def save_anndata(
    adata: ad.AnnData, prefix: str, output_dir: PurePath, pval_max: float = 0.01
):

    ad_dir = output_dir / f"{prefix}_anndata"
    ad_dir.mkdir(parents=False, exist_ok=True)

    adata.write_h5ad(ad_dir / "adata.h5ad")
    print("adata saved.")

    try:
        sc.pl.umap(
            adata,
            color="leiden",
            legend_loc="on data",
            show=False,
        ).figure.savefig(ad_dir / "knn_umap_leiden.pdf")
        print("umap saved.")
    except KeyError:
        print("no leiden")

    try:
        rgg = adata.uns["rank_genes_groups"]
        cluster = len(adata.uns["leiden_colors"])
        temp_pdf_list = []
        for c in range(cluster):
            temp_pdf = pl.DataFrame(
                data={
                    "genes": rgg["names"][str(c)],
                    "cluster": [c] * len(rgg["names"][str(c)]),
                    "pvals": rgg["pvals_adj"][str(c)],
                    "logfc": rgg["logfoldchanges"][str(c)],
                }
            ).sort("pvals")
            temp_pdf_list.append(temp_pdf.filter(pl.col("pvals") < pval_max))

        pdf = pl.concat(temp_pdf_list)
        pdf.write_csv(ad_dir / "ranked_genes.csv")
        print("ranked csv saved.")
    except KeyError:
        print("no ranking data.")

    try:
        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=3,
            values_to_plot="logfoldchanges",
            cmap="bwr",
            colorbar_title="log fold change",
            show=False,
            figsize=(12, 6),
        )["mainplot_ax"].figure.savefig(ad_dir / "ranked_genes_dotplot.pdf")
        print("ranked dotplot saved.")
    except TypeError:
        print("no ranking data.")

    try:
        sq.pl.spatial_scatter(
            adata,
            shape=None,
            color="leiden",
            size=50,
            figsize=(50, 50),
            save=(ad_dir / "spatial_scatter_leiden.pdf"),
        )
        print("spatial scatter saved")
    except KeyError:
        print("no spatial data or leiden")