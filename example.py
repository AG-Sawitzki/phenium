import os
from pathlib import Path

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "16"

from rich.traceback import install

install(show_locals=True)

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from phenium import analysis, io, plot
from phenium.main import Phenium
from phenium.utils import pixelate

xenium_dir = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260827__133139__20260827_ITCAIM_lung-spleen_run4/output-XETG00410__0065619__non_infected_lung__20260827__133207"
pheno_img = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260902_ITCAIM_Phenium_Slide8/Scan1/20260902_ITCAIM_Phenium_Slide8_Scan1.er.qptiff"
output_dir = "/data/cephfs-1/scratch/projects/sawitzki-phenium/Phenium_XOA-prior"
config_path = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/phenium/config.toml"
prior_type = "XOA"
channels = ("DAPI", "CD44", "Vimentin")
exclude = [
    "CD69",
    "CD206",
    "TCRgd",
    "CD16",
    "Osteopontin",
    "FoxP3",
    "CD11c",
    "CD36",
    "CD62L",
    "TCRab",
]
seed = 37

ana = Phenium(
    SRT_dir=xenium_dir,
    IHC_path=pheno_img,
    output_dir=output_dir,
    config_path=config_path,
    prior_type=prior_type,
    channels=channels,
    exclude=exclude,
    seed=seed
)

ana.prepare(4, 12, quick=True, verbose=True)

adata = ana.IHC.cluster(
    resolution=0.175,
    exclude=["DAPI", "CD44", "Vimentin"],
    mask=ana.masks,
    return_copy=True,
)
io.save_anndata(adata, prefix="IHC_wXOA", output_dir=ana.output_dir)
fig, ax = ana.IHC.plot_segmentation(plgns=ana.masks, channels=channels)
fig.savefig(ana.output_dir / "plots/IHC_XOA_seg_on_IHC-DAPI-CD44.pdf")

ana.IHC.segment(channels=channels)
ana.IHC.cluster(resolution=0.175, exclude=["DAPI", "CD44", "Vimentin"])
io.save_anndata(ana.IHC.adata, prefix="IHC_wCP", output_dir=ana.output_dir)
fig, ax = ana.IHC.plot_segmentation(channels=channels)
fig.savefig(ana.output_dir / 'plots/IHC_CP_seg_on_IHC-DAPI.pdf')

ana.run_proseg()

masks = io.load_plgns(ana.output_dir / "cell-polygons.geojson.gz")
masks = pixelate(masks, ana.XMI.pxsize[0])

fig, ax = plt.subplots()
fig, ax = plot.polygon_overlay(
    masks, ana.XMI.roi[0, :, :], fig, ax, ana.XMI.pxsize[0]
)
fig.savefig(ana.output_dir / "plots/proseg_on_DAPI.pdf")

ana.cluster(resolution=0.175, save=True)

vrlp = analysis.ovrlpy_analysis(ana.IST.roi, "xenium")

doublets = vrlp.detect_doublets(min_signal=3, integrity_sigma=2)

doublets = pixelate(doublets, ana.XMI.pxsize[0])

masks = io.load_plgns(Path(output_dir) / "cell-polygon-layers.geojson.gz")
masks = pixelate(masks, ana.XMI.pxsize[0])

success_rate, stuff = analysis.vis_doublets(
    doublets,
    masks,
)

other_stuff = analysis.proseg_doublets(masks)

fig, ax = plt.subplots()
fig, ax = plot.polygon_overlay(
    masks[masks.layer == 0],
    ana.XMI.roi[0, :, :],
    fig, ax,
    ana.XMI.pxsize[0]
)
fig, ax = plot.scatter_doublets(stuff, fig, ax, doublets)
fig.savefig(ana.output_dir / "plots/proseg_L0_on_DAPI_wDoublets.pdf")

cluster_celltype_dict = {
    "0": "endothel",
    "1": "immune cell",
    "2": "immune cell",
    '3': 'epithel',
    "4": "epithel",
    "5": "epithel",
}
celltype_prot_dict = {
    "epithel": [
        "DAPI",
        "CD44",
        "Vimentin",
        "Epcam",
        "PanCK",
        "S100A9",
        "Caveolin",
        "CD31",
        "ITGB1",
    ],
    "immune cell": [
        "DAPI",
        "CD44",
        "Vimentin",
        "MHC II",
        "CD31",
        "CD45",
        "CD19",
        "CD20",
        "CD45R/B220",
        "CD68",
        "CD11b",
        "Iba1",
        "F4/80",
        "S100A9",
        "CD31",
        "ITGB1",
        "Ki67",
        "Ly6g",
        "CD3e",
        "CD8",
        "CD4",
    ],
    "endothel": ["DAPI", "CD44", "Vimentin", "CD31", "ITGB1"],
}

new_img = ana.deconvolute(
    None,
    None,
    img=ana.XMI.roi,
    biomarkers=ana.XMI.roi_biomarkers,
)

low = int(11652 / 2 - 500)
high = int(11652 / 2 + 500)

tifffile.imwrite(
    ana.output_dir / "decon_img_XMI.tif",
    new_img[:, :, low:high, low:high],
    bigtiff=True,
    metadata={"axes": "ZCYX"},
)

fctrs = np.ones_like(ana.XMI.roi, dtype=np.float64)
np.divide(
    new_img.sum(axis=0).astype(np.float64),
    ana.XMI.roi.astype(np.float64),
    out=fctrs,
    where=True,
)
np.nan_to_num(fctrs, copy=False, nan=1)
np.divide(new_img, fctrs, out=new_img, where=True)

tifffile.imwrite(
    ana.output_dir / "decon_img_XMI_wfctrs.tif",
    new_img[:, :, low:high, low:high],
    bigtiff=True,
    metadata={"axes": 'ZCYX'},
)

del new_img

new_img = ana.deconvolute(
    cluster_celltype_dict,
    celltype_prot_dict,
    img=ana.IHC.roi,
    biomarkers=ana.IHC.roi_biomarkers,
)

channels = [ana.IHC.roi_biomarkers[c] for c in channels]

tifffile.imwrite(
    ana.output_dir / "decon_img_IHC.tif",
    new_img[:, channels, low:high, low:high],
    bigtiff=True,
    metadata={"axes": "ZCYX"},
)

fctrs = np.ones_like(ana.IHC.roi[channels, :, :], dtype=np.float64)
np.divide(
    new_img.sum(axis=0).astype(np.float64),
    ana.IHC.roi.astype(np.float64),
    out=fctrs,
    where=True,
)
np.nan_to_num(fctrs, copy=False, nan=1)
np.divide(new_img, fctrs, out=new_img, where=True)


tifffile.imwrite(
    ana.output_dir / "decon_img_IHC_wfctrs.tif",
    new_img[:, channels, low:high, low:high],
    bigtiff=True,
    metadata={"axes": "ZCYX"},
)
