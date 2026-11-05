import os
import timeit
from pathlib import Path

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "16"

from rich.traceback import install

install(show_locals=True)

import tifffile
from phenium import analysis, io
from phenium.main import Phenium

xenium_dir = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260827__133139__20260827_ITCAIM_lung-spleen_run4/output-XETG00410__0065619__non_infected_lung__20260827__133207"
pheno_img = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260902_ITCAIM_Phenium_Slide8/Scan1/20260902_ITCAIM_Phenium_Slide8_Scan1.er.qptiff"
output_dir = "/data/cephfs-1/scratch/projects/sawitzki-phenium"
config_path = "/data/cephfs-2/unmirrored/projects/sawitzki-phenium/phenium/config.toml"
prior_type = "CP"
channels = ("DAPI", "CD45")
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

ana = Phenium(
    SRT_dir=xenium_dir,
    IHC_path=pheno_img,
    output_dir=output_dir,
    config_path=config_path,
    prior_type=prior_type,
    channels=channels,
    exclude=exclude
)

ana.prepare(4, 12, verbose=True)

ana.IHC.cluster(resolution=0.175, exclude=["DAPI"])
io.save_anndata(ana.IHC.adata, prefix="IHC", output_dir=Path(output_dir))
ana.IHC.find_cell_contour(verbose=True)
fig, ax = ana.IHC.plot_segmentation(channels=channels)
fig.savefig(ana.output_dir / 'IHC_CP_seg_on_IHC-DAPI.pdf')

# if prior_type != "CP":
#     ana.IHC.segment()
#     ana.IHC.cluster(resolution=0.175, exclude=exclude)
#     io.save_anndata(ana.IHC.adata, prefix="IHC", output_dir=Path(output_dir))
# else:
#     if ana.CP_params['style'] == 'pheno':
#         ana.IHC.cluster(resolution=0.175, exclude=exclude)
#         io.save_anndata(ana.IHC.adata, prefix="IHC", output_dir=Path(output_dir))

ana.run_proseg()

ana.cluster(resolution=0.175, save=True)

vrlp = analysis.ovrlpy_analysis(
    ana.IST.roi, 'xenium'
)

doublets = vrlp.detect_doublets(
    min_signal=3,
    integrity_sigma=2
)

masks = io.load_plgns(Path(output_dir) / "cell-polygon-layers.geojson.gz")

success_rate, stuff = analysis.vis_doublets(
    doublets,
    masks,
)

other_stuff = analysis.proseg_doublets(masks)

new_img = ana.function_name(None, None)

low = int(11652 / 2 - 500)
high = int(11652 / 2 + 500)

tifffile.imwrite(
    output_dir / "func_name_img_IHC.tif",
    new_img[:, :, low:high, low:high],
    bigtiff=True,
)
