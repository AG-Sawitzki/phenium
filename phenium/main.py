from phenium.utils.image_processing import find_rois

import multiprocessing as mp
from pathlib import Path

import tomlkit
import tifffile
import ovrlpy
import numpy
import polars
import zarr


class NotImplemented(Exception):
    pass


class Phenium(object):
    def __init__(
            self,
            input_dir,
            output_dir,
            config_path,
        )-> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)

    def _parse_config(self):
        with open(self.config_path) as f:
            self.config = tomlkit.load(f)

    def _load_as_zarr(self, focus=False):
        if focus:
            raise NotImplementedError(
                'Nor Implemented to load morphology_focus as zarr.'
            )
        else:
            img_path = self.input_dir / 'morphology.ome.tif'
        img_store = tifffile.imread(
            img_path,
            aszarr=True
        )
        self.img_zarr = zarr.open(img_store, mode='r')

    def find_sections(self, focus=False):
        self._load_as_zarr(focus=focus)
        subresolution_level = [lvl for lvl in self.img_zarr]
        # isn't the above basically img_zarr.len()??
        # if yes then set self.img_lvl = max and the loop through range
        self.img_shape = self.img_zarr[min(subresolution_level)].shape
        contours, _ = find_rois(self.img_shape, self.img_zarr[max(subresolution_level)], 0)


    def _transcripts_for_ovrlpy(self):
        machine = self.config['type']
        if machine == 'xenium':
            self.coordinates = ovrlpy.io.read_Xenium(
                filepath=self.input_dir / 'transcripts.parquet',
                min_qv=None,
                remove_features=ovrlpy.io.XENIUM_CTRLS,
                additional_columns=None,
                n_threads=mp.cpu_count(),
            )
        else:
            raise NotImplementedError(f"Not Implemented to deal with `{machine}`.")
