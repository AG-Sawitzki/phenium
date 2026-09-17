from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
    get_affine_matrix,
    get_pixel_size,
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
            SRT_dir,
            IHC_path,
            output_dir,
            config_path,
        )-> None:
        self.SRT_dir = Path(SRT_dir)
        self.IHC_path = Path(IHC_path)
        self.output_dir = Path(output_dir)
        self.config_path = Path(config_path)

    def _parse_config(self):
        with open(self.config_path) as f:
            config = tomlkit.load(f)

        self.machine = config['Basics']['machine']
        self.priors = config['Basics']['priors']

        self.CP_params = config['cellpose']
        self.PS_params = config['proseg']

    def _load_imgs(self):
        # load morphology_focus image
        SRT_store = tifffile.imread(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif',
            aszarr=True
        )
        self.SRT_zarr = zarr.open(SRT_store, mode='r')
        self.SRT_shape = self.SRT_zarr['0'].shape

        self.SRT_pxsize = get_pixel_size(
            self.SRT_dir / 'morphology_focus/ch0000_dapi.ome.tif'
        )

        # load phenocycler image
        IHC_store = tifffile.imread(
            self.IHC_path,
            aszarr=True
        )
        self.IHC_zarr = zarr.open(IHC_store, mode='r')
        self.IHC_shape = self.IHC_zarr['0'].shape



    def _affine_matrix(self):
        matrix = get_affine_matrix(..., ...)
        return matrix

    def _find_sections(self):
        subresolution_lvl = len(self.SRT_zarr)
        bboxes = find_rois(
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-1)]
        )
        self.sections = bboxes_to_sections(
            bboxes,
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-1)].shape
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

    def homographyMatrix(self):
        ...
        return self.hMatrix

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

        if style == 'focus':
            zrr = self.SRT_zarr
        if style == 'pheno':
            zrr = self.IHC_zarr
        
        coords = self.sections[section]
        img = zrr['0'][
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
        self._load_imgs()
        self._find_sections()
        self.homographyMatrix()

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
        elif self.priors == 'XOA':
            self._XOA_masks(
                section
            )
            self.prior_path = self.output_dir / f'XOA_masks_{section}.npy'
        else:
            self.prior_path = ''

        print(
            f' Priors set to {self.priors}.'
            f'\n Prior_path: {self.prior_path}'
        )

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
        self.counts = fmm.mmread(counts_path, parallelism=mp.cpu_count())

    # def run_leiden():

    # def assign_CT_probabilities():

    # def assign_stain_to_CT():

    # def assign_pixel_to_cell():

    # def {function for running cluster_CellTypes, assign_CT, stain_to_Cell}