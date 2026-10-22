import matplotlib.pyplot as plt
import ovrlpy
from shapely import Point

from phenium.exceptions import NotImplemented
from phenium.utils import transcripts_for_ovrlpy


def ovrlpy_analysis(transcripts, machine, **kwargs):
    cdf = transcripts_for_ovrlpy(
        machine=machine,
        transcripts=transcripts,
    )

    ovrlpy_dataset = ovrlpy.Ovrlp(
        cdf,
        # KDE_bandwidth,
        # min_distance,
        # n_components,
        # gene_key,
        # coordinate_keys,
        # n_workers,
        # dtype,
        # patch_length,
        # umap_kwargs,
        # cumap_kwargs,
        # random_state,
    )

    # should be split into the processes
    # to include all params for
    # process_coordinates
    # fit_transcripts
    # compute_VSI
    ovrlpy_dataset.analyse(
        # gridsize,
        # min_transcripts,
        # genes,
        # fit_umap,
    )

    # if plot:
    #     fig = ovrlpy.plot_signal_integrity(
    #         ovrlpy_dataset,
    #         signal_threshold=4,
    #     )
    #     fig.savefig(self.output_dir / "plots/signal_integrity_map.pdf")

    #     fig = ovrlpy.plot_pseudocells(ovrlpy_dataset)
    #     fig.savefig(self.output_dir / "plots/pseudocells.pdf")

    return ovrlpy_dataset


def vis_doublets(ovrlpy_dataset, masks):
    doublets = ovrlpy_dataset.detect_doublets(
        min_signal=3,
        integrity_sigma=2,
    )

    # compare masks and doublets
    stuff = {}
    for idx, row in enumerate(doublets.iter_rows(named=True)):
        pnkt = Point(row["x"], row["y"]) # perhaps not a point but a small square around that point?
        fltr = masks["geometry"].covers(pnkt)
        if fltr.any():
            plgns = masks[fltr]
            cells = plgns.cell.unique()
            if cells.size > 1:
                stuff["double_cell"][str(idx)] = cells
            else:
                stuff["single_cell"][str(idx)] = cells

    success_rate = len(stuff["double_cell"]) / len(doublets)

    fig, ax = plt.subplots()
    _scatter = ax.scatter(
        doublets["x"],
        doublets["y"],
        c=doublets["integrity"],
        s=0.5,
        cmap="viridis",
    )

    _ = ax.set_aspect("equal")
    _ = fig.colorbar(_scatter, ax=ax)

    return success_rate


def verify_doublets(adata, stuff):
    pass
    # for each doublet
        # identify celltypes
        # compare with ovrlpy celltypes?
        # if two different cells ovrelap then 
            # check if off different celltype
