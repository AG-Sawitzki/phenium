from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
    load_counts,
    get_affine_matrix,
)
from phenium.exceptions import (
    NotImplemented
)
from phenium.segment import (
    cellpose_wrapper,
    proseg_wrapper
)

import xml.etree.ElementTree as ET
import multiprocessing as mp
import json
from pathlib import (
    PosixPath,
    Path
)
from typing import Union

import fast_matrix_market as fmm
import numpy as np
import polars as pl
import tomlkit
import tifffile
import ovrlpy
import zarr


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
            config = tomlkit.load(f)

        basics = config['Basics']
        self.machine = basics['machine']
        self.priors = basics['priors']
        self.pixelsizeXY_SRT = basics['pixelsize_xy']
        self.pixelsizeZ_SRT = basics['pixelsize_z']

        self.CP_params = config['cellpose']
        self.PS_params = config['proseg']

    def _load_imgs(self):
        # only load PhenoCycler or also morphology?
        img_store = tifffile.imread(
            self.input_dir / 'PhenoCycler_img.qptiff',
            aszarr=True
        )
        self.img_zarr = zarr.open(img_store, mode='r')
        self.img_shape = self.img_zarr['0'].shape

    def _affine_matrix(self):
        # find warp-matrix
        im1_store = tifffile.imread('...', aszarr=True)
        im1_zarr = zarr.open(im1_store, mode='r')
        im2_store = tifffile.imread('...', aszarr=True)
        im2_zarr = zarr.open(im2_store, mode='r')

        im1_arr = np.array(im1_zarr[str(len(im1_zarr)-1)][5], dtype=np.uint8)
        im2_arr = np.array(im2_zarr[str(len(im2_zarr)-1)][0], dtype=np.uint8)
        matrix = get_affine_matrix(im1_arr, im2_arr)
        return matrix

    def _find_sections(self):
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

    def _get_shape(self, section):
        coords = self.sections[section]
        x = coords['x'][1]-coords['x'][0]
        y = coords['y'][1]-coords['y'][0]
        return x, y

    def _filter_for_section(
            self,
            table,
            section,
        ) -> pl.dataframe.frame.DataFrame:
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        coords = self.sections[section]
        filtered_table = filter_by_location(
            table,
            coords,
            pixelsize=self.pixelsizeXY_SRT
        )

        return filtered_table

    def _XOA_masks(self, section):
        path = self.input_dir / 'cell_boundaries.parquet'
        table = pl.read_parquet(path)
        table = self.filter_for_section(section, table)
        # shape = self.get_shape(section)
        masks, polygons = boundaries_to_mask(table, self.roi_shape)
        np.save(
            self.output_dir / f'XOA_masks_{section}.npy',
            masks,
            allow_pickle=True
        )
        with open(self.output_dir / f'XOA_polygons_{section}.json') as f:
            json.dump(f, polygons)

    def _run_cellpose(self, section):
        model_params = self.CP_params['model']
        eval_params = self.CP_params['eval']
        style = self.CP_params['style']

        if style == 'pheno':
            path = self.input_dir / '...'
        if style == 'focus':
            path = self.input_dir / '---'
        
        self._load_img(path)
        coords = self.sections[section]
        img = self.img_zarr['0'][
            ...,
            coords['x'][0]:coords['x'][1],
            coords['y'][1]:coords['y'][1]
        ]

        masks, flows, styles = cellpose_wrapper(
            img,
            model_params,
            eval_params,
            style,
        )

        np.save(
            self.output_dir / f'CP_masks_{section}.npy',
            masks,
            allow_pickle=True
        )

    def prepare(self, section):
        '''
        Prepare `Phenium` and the input for the Segmentation and Deconvolution.
        '''
        self._parse_config()

        self._find_sections()

        if section in self.sections:
            self.section = section
        else:
            # this error isn't returned
            KeyError(f'This section (`{section}`) does not exist.')
            return None

        self.roi_shape = self._get_shape(section)
    
        if self.priors == 'CP':
            self._run_cellpose(
                section
            )
            self.prior_path = self.output_dir / f'CP_masks_{section}.npy'
        if self.priors == 'XOA':
            self._XOA_masks(
                section
            )
            self.prior_path = self.output_dir / f'XOA_masks_{section}.npy'
        if self.priors == 'None':
            self.prior_path = ''

        transcripts = pl.read_parquet(self.input_dir / 'transcripts.parquet')
        transcripts = self._filter_for_section(
            transcripts,
            self.section
        )

        self.transcripts_path = (
            self.output_dir / f'transcripts_{section}.parquet'
        )
        transcripts.write_parquet(
            self.transcripts_path,
            compression='snappy',
            use_pyarrow=True,
            pyarrow_options=dict(
                version = '1.0',
            )
        )

    def get_VSI(self, transcripts_path, **kwargs):
        cdf = transcripts_for_ovrlpy(
            machine=self.machine,
            file_path=transcripts_path,
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

        return ovrlpy_dataset.integrity_map, ovrlpy_dataset.signal_map

    def _run_proseg(
        self,
    ) -> None:

        proseg_wrapper(
            self.PS_params,
            self.transcripts_path,
            self.output_dir,
            self.pixelsizeXY_SRT,
            self.machine,
            self.prior_path
        )


    # ... more functions

    # def {function for running proseg, get_VSI, compare_VSI}
    #   assign transcripts to layers
    # 
    #   get_VSI and compare_VSI must be optional...
    #   hence maybe separate into two functions

    # ... more functions
    def _load_counts(self):
        counts_path = self.output_path / 'counts.mtx.gz'
        self.counts = load_counts(counts_path)

    # def run_leiden():

    # def assign_CT_probabilities():

    # def assign_stain_to_CT():

    # def assign_pixel_to_cell():

    # def {function for running cluster_CellTypes, assign_CT, stain_to_Cell}