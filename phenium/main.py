from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    to_cli,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
)
from phenium.exceptions import (
    NotImplemented
)
from phenium.segment import (
    cellpose_wrapper,
    proseg_wrapper
)

from subprocess import Popen
import multiprocessing as mp
from pathlib import Path
from typing import Union

import numpy as np
import tomlkit
import tifffile
import ovrlpy
import polars
import zarr

PDF = polars.dataframe.frame.DataFrame


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
        self._load_config()
        self.machine = self.config['Basics']['type']
        self.pixelsizeXY_SRT = self.config['Basics']['pixelsize_xy']
        self.pixelsizeZ_SRT = self.config['Basics']['pixelsize_z']

    def _load_config(self):
        with open(self.config_path) as f:
            self.config = tomlkit.load(f)

    def _load_img(self, img_path, focus=False, pheno=False):
        if not img_path:
            if focus or pheno:
                raise NotImplementedError(
                    'Not Implemented to load focus or pheno image as zarr.'
                )
            else:
                img_path = self.input_dir / 'morphology.ome.tif'

        img_store = tifffile.imread(
            img_path,
            aszarr=True
        )

        self.img_zarr = zarr.open(img_store, mode='r')
        self.img_shape = self.img_zarr['0'].shape

    def find_sections(self):
        self._load_img()
        subresolution_lvl = len(self.img_zarr)
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

    def filter_for_section(
            self,
            section,
            table,
        ) -> polars.dataframe.frame.DataFrame:
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        coords = self.sections[str(section)]
        filtered_table = filter_by_location(
            self.transcripts,
            coords
        )

        return filtered_table

    def get_VSI(self):
        cdf = transcripts_for_ovrlpy(
            machine=self.config['Basics'],
            file_path=self.input / 'transcripts.parquet',
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
        # compute_VSI()
        ovrlpy_dataset.analyse(
            # gridsize,
            # min_transcripts,
            # genes,
            # fit_umap,
        )

        return ovrlpy_dataset.integrity_map, ovrlpy_dataset.signal_map

    def _XOA_masks(self):
        paths = list(self.input_path.glob('*_boundaries.parquet'))
        for path in paths:
            table = polars.read_parquet()
            masks, polygons = boundaries_to_mask(table, self.img_shape[1:])
        self.XOA = masks
        np.save(
            self.output_path / 'XOA_masks.npy',
            allow_pickle=True
        )
        return None

    def run_cellpose(self):
        params = self.config['cellpose']
        model_params = params['model']
        eval_params = params['eval']
        style = params['style']

        masks, flows, styles = cellpose_wrapper(
            model_params,
            eval_params,
            style,
        )

        return None

    def _run_proseg(
        self,
        section,
        XOA: bool = False,
        CP: bool = False,
    ) -> None:
        transcripts_path = self.output_path / f'transcripts_{section}.parquet'

        if XOA and not CP:
            prior_path = self.output_path / f'XOA_masks_{section}.npy'
        if CP and not XOA:
            prior_path = self.output_path / f'CP_masks_{section}.npy'

        proseg_wrapper(
            self.config['proseg'],
            transcripts_path,
            self.output_dir,
            prior_path
        )

        return None
