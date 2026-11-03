from pathlib import PurePath

import anndata as ad
import geopandas as gpd
import polars as pl
import scanpy as sc
import squidpy as sq

from phenium.exceptions import NotImplemented


def load_plgns(path):
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
        ).figure.savefig(ad_dir / "plots/knn_umap_leiden.pdf")
        print("umap saved.")
    except:
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
    except:
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
    except:
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
    except:
        print("no spatial data.")