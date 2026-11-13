import ovrlpy
from shapely import Point, area, intersection

from phenium.plot import spatial_transcripts, transcripts_vs_z, z_distribution
from phenium.utils import transcripts_for_ovrlpy


def z_spread(table):
    ax = spatial_transcripts(table)
    ax = transcripts_vs_z(table)
    ax = z_distribution(table)
    return ax


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


def vis_doublets(doublets, masks):
    # doublets = ovrlpy_dataset.detect_doublets(
    #     min_signal=3,
    #     integrity_sigma=2,
    # )

    # compare masks and doublets
    ovrlpy_doublets = {"no_cell": {}, "single_cell": {}, "double_cell": {}}
    for idx, row in enumerate(doublets.iter_rows(named=True)):
        pnkt = Point(row["x"], row["y"]) # perhaps not a point but a small square around that point?
        fltr = masks["geometry"].covers(pnkt)
        if fltr.any():
            plgns = masks[fltr]
            cells = plgns.cell.unique()
            if cells.size > 1:
                ovrlpy_doublets["double_cell"][str(idx)] = cells
            else:
                ovrlpy_doublets["single_cell"][str(idx)] = cells
        else:
            ovrlpy_doublets["no_cell"][str(idx)] = pnkt

    success_rate = len(ovrlpy_doublets["double_cell"]) / len(doublets)

    # fig, ax = plt.subplots()
    # _scatter = ax.scatter(
    #     doublets["x"],
    #     doublets["y"],
    #     c=doublets["integrity"],
    #     s=0.5,
    #     cmap="viridis",
    # )

    # _ = ax.set_aspect("equal")
    # _ = fig.colorbar(_scatter, ax=ax)

    return success_rate, ovrlpy_doublets


def verify_doublets(adata, ovrlpy_doublets):
    pass
    # for each doublet
        # identify celltypes
        # compare with ovrlpy celltypes?
        # if two different cells ovrelap then 
            # check if off different celltype


def foo(row, gdf, midx):
    idx, (_oidx, cell, layer, geom) = row
    low = max(0, idx-25)
    high = min(idx+25, midx)
    gdf = gdf.iloc[low:high]
    fltr = gdf.intersects(geom)
    vrlps = gdf[fltr]
    cell_fltr = vrlps.cell != cell
    doublets = vrlps[cell_fltr]
    return {cell: {layer: list(doublets.cell)}}


def proseg_doublets(gdf, proseg_doublets):
    proseg_doublets = {}
    sjnd = gdf.sjoin(gdf, how='inner', predicate='intersects')
    vrlps = sjnd[sjnd.cell_left != sjnd.cell_right]
    cids = set(vrlps.cell_left)
    for cid in cids:
        proseg_doublets[cid] = list(set(vrlps[vrlps.cell_left == cid].cell_right))
    return proseg_doublets
    # for cell check if any layer overlaps with any other cell


def IoU(plgn1, plgn2, gdf):
    I = area(intersection(plgn1, plgn2))
    IoU = I / (area(plgn1) + area(plgn2) - I)
    return IoU


def doublet_size(gdf):
    sjnd = gdf.sjoin(gdf, how='inner', predicate='intersects')
    vrlps = sjnd[sjnd.cell_left != sjnd.cell_right]
    vrlps['geometry_right'] = gdf.geometry[vrlps.index_right].to_list()
    vrlps['IoU'] = vrlps.apply(
        lambda x: IoU(x.geometry, x.geometry_right, gdf), axis=1
    )
    return vrlps
