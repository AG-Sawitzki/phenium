from phenium.main import Phenium
import spatialdata

obj = Phenium(
    SRT_dir='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260716__161300__20260716_ITCAIM_lung-spleen_run1/output-XETG00410__0064543__lung__20260716__161319',
    IHC_path='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/data/20260723_ITCAIM_Phenium_Slide1/Scan1/20260723_ITCAIM_Phenium_Slide1_Scan1.er.qptiff',
    output_dir='/data/cephfs-1/scratch/projects/sawitzki-phenium',
    config_path='/data/cephfs-2/unmirrored/projects/sawitzki-phenium/phenium/config.toml',
)

obj.prepare(4,6, quick=True, verbose=True)

obj._run_proseg()

sdata = spatialdata.read_zarr(obj.output_dir / 'proseg-output.zarr')