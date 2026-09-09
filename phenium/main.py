from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location
)

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
        # load boundaries here? or in func?
        # same for img...

    def _parse_config(self):
        with open(self.config_path) as f:
            self.config = tomlkit.load(f)

    def _load_as_zarr(self, focus=False):
        if focus:
            raise NotImplementedError(
                'Not Implemented to load morphology_focus as zarr.'
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
        subresolution_lvl = len(self.img_zarr)
        self.img_shape = self.img_zarr['0'].shape
        bboxes = find_rois(
            self.img_shape,
            self.img_zarr[str(subresolution_lvl-1)]
        )
        self.sections = bboxes_to_sections(
            bboxes,
            self.img_shape,
            self.img_zarr[str(subresolution_lvl-1)].shape
        )
        return self.sections

    def _transcripts_for_ovrlpy(self):
        machine = self.config['type']
        if machine == 'xenium':
            self.transcripts = ovrlpy.io.read_Xenium(
                filepath=self.input_dir / 'transcripts.parquet',
                min_qv=None,
                remove_features=ovrlpy.io.XENIUM_CTRLS,
                additional_columns=['transcript_id'],
                n_threads=mp.cpu_count(),
            )
        else:
            raise NotImplementedError(
                f"Not Implemented to deal with `{machine}`."
            )

    # def load_boundaries?

    def filter_for_section(self, table, section):
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        coords = self.sections[str(section)]
        filtered_table = filter_by_location(
            self.transcripts,
            coords
        )

        return filtered_table

    # def compute_VSI?