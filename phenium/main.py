from phenium.utils import (
    find_rois,
    bboxes_to_sections,
    filter_by_location,
    boundaries_to_mask,
    transcripts_for_ovrlpy,
    # find_transform,
    get_pixel_size,
)
from phenium.exceptions import (
    NotImplemented
)
from phenium.segment import (
    cellpose_wrapper,
    proseg_wrapper
)

import multiprocessing as mp
import os
import json
from pathlib import (
    PosixPath,
    Path
)
from typing import Union, Tuple

# import fast_matrix_market as fmm
import numpy as np
import polars as pl
from scipy.io import loadmat, mmread
from skimage.transform import rescale
import ants
import tifffile
import tomlkit
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

        self._parse_config()
        self.load_imgs()
        self.find_sections()

        print('Initialized. Review marked images and provide sections.')

    def _parse_config(
        self,
    ) -> None:
        with open(self.config_path) as f:
            config = tomlkit.load(f)

        self.machine = config['Basics']['machine']
        self.priors = config['Basics']['priors']

        self.CP_params = config['cellpose']
        self.PS_params = config['proseg']

    def load_imgs(
        self,
    ) -> None:
        '''
        Load the SRT and IHC images as zarr-storages, retrieves the pixelsize,
        and add the shape to the object variables.
        '''
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

        self.IHC_pxsize = get_pixel_size(
            self.IHC_path
        )

        self.scaling_factor = self.IHC_pxsize[0]/self.SRT_pxsize[0]

    def find_sections(
        self
    ) -> None:
        subresolution_lvl = len(self.SRT_zarr)
        bboxes = find_rois(
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-2)]
        )
        self.SRT_sections, marked_img = bboxes_to_sections(
            bboxes,
            self.SRT_shape,
            self.SRT_zarr[str(subresolution_lvl-2)]
        )

        tifffile.imwrite(
            self.output_dir / 'marked_regions_SRT.tif',
            marked_img
        )

        subresolution_lvl = len(self.IHC_zarr)
        bboxes = find_rois(
            self.IHC_shape,
            self.IHC_zarr[str(subresolution_lvl-1)]
        )
        self.IHC_sections, marked_img = bboxes_to_sections(
            bboxes,
            self.IHC_shape,
            self.IHC_zarr[str(subresolution_lvl-1)]
        )

        tifffile.imwrite(
            self.output_dir / 'marked_regions_IHC.tif',
            marked_img
        )

    def _get_shape(
        self,
    ) -> Tuple[int, int]:
        x = self.SRT_coords['x'][1]-self.SRT_coords['x'][0]
        y = self.SRT_coords['y'][1]-self.SRT_coords['y'][0]

        return x, y

    def _filter_for_section(
        self,
        table,
        section,
    ) -> pl.dataframe.frame.DataFrame:
        # must be adapted for boundaries!
        # currently only transcripts are passed...
        filtered_table = filter_by_location(
            table,
            self.SRT_coords,
            pixelsize=self.SRT_pxsize[0]
        )

        return filtered_table

    def find_affine(
        self,
        quick=False,
        verbose=False
    ) -> None:
        SRT_img = self.SRT_img[0, :, :]
        IHC_img = self.IHC_zarr['0'][
            0,
            self.IHC_coords['y'][0]:self.IHC_coords['y'][1],
            self.IHC_coords['x'][0]:self.IHC_coords['x'][1]
        ]
        IHC_img = rescale(
            IHC_img,
            self.scaling_factor,
            preserve_range=True,
        )
        IHC_img = np.flip(IHC_img)

        SRT_img = np.uint32(SRT_img)
        IHC_img = np.uint32(IHC_img)

        static = ants.from_numpy(SRT_img)
        moving = ants.from_numpy(IHC_img)

        if quick:
            tof = 'antsRegistrationSyNQuick[r]'
        else:
            tof = 'antsRegistrationSyN[r]'

        reg = ants.registration(
            fixed=static,
            moving=moving,
            type_of_transform=tof,
            verbose=verbose,
        )

        mat = loadmat(reg['fwdtransforms'][0])
        keys = list(mat.keys())

        dim = SRT_img.ndim
        A = mat[keys[0]][:dim**2].reshape((dim, dim))
        t = mat[keys[0]][dim**2:]
        c = mat[keys[1]]

        w = t + c - np.dot(A, c)
        Ah = np.identity(dim+1)
        Ah[:dim, :dim] = A
        Ah[:dim, dim:] = w

        self.affine_transform = Ah
        self.reg_fwd = reg['fwdtransforms']

    def _transform_IHC_channel(self, channel):
        img = self.IHC_zarr['0'][
            channel,
            self.IHC_coords['y'][0]:self.IHC_coords['y'][1],
            self.IHC_coords['x'][0]:self.IHC_coords['x'][1]
        ]
        transformed_img = ants.apply_transforms(
            fixed=self.SRT_img[0],
            moving=img,
            transformlist=self.reg_fwd,
            interpolator='nearestNeighbor',
        )

        return transformed_img

    def _XOA_masks(self, section):
        path = self.SRT_dir / 'cell_boundaries.parquet'
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

    def _run_cellpose(self):
        model_params = self.CP_params['model']
        eval_params = self.CP_params['eval']
        style = self.CP_params['style']
        channels = self.CP_params['channels']

        if style == 'focus':
            zrr = self.SRT_zarr
            img = zrr['0'][
                channels,
                self.SRT_coords['y'][0]:self.SRT_coords['y'][1],
                self.SRT_coords['x'][0]:self.SRT_coords['x'][1]
            ]
        if style == 'pheno':
            imgs = []
            for c in channels:
                imgs.append(self._transform_IHC_channel(c))
            img = np.dstack(imgs)
            img = np.moveaxis(img, -1, 0)

        masks, flows, styles = cellpose_wrapper(
            img,
            model_params,
            eval_params,
            style,
        )

        np.save(
            self.output_dir / f'CP_masks_{self.SRT_section}.npy',
            masks,
            allow_pickle=True
        )

    def prepare(self, SRT_section, IHC_section, quick=True, verbose=True):
        '''
        Prepare `Phenium` and the input for the Segmentation and Deconvolution.
        '''
        if (
            (SRT_section in self.SRT_sections.keys())
            and
            (IHC_section in self.IHC_sections.keys())
        ):
            self.SRT_section = SRT_section
            self.IHC_section = IHC_section
        else:
            # this error isn't returned
            KeyError(f'This section (`{SRT_section}`) does not exist.')
            return None

        self.IHC_coords = self.IHC_sections[self.IHC_section]
        self.SRT_coords = self.SRT_sections[self.SRT_section]

        self.SRT_img = self.SRT_zarr['0'][
            :,
            self.SRT_coords['y'][0]:self.SRT_coords['y'][1],
            self.SRT_coords['x'][0]:self.SRT_coords['x'][1]
        ]

        self.find_affine(quick=quick, verbose=verbose)

        self.roi_shape = self._get_shape()

        if self.priors == 'CP':
            self._run_cellpose(
                SRT_section
            )
            self.prior_path = self.output_dir / f'CP_masks_{SRT_section}.npy'
        elif self.priors == 'XOA':
            self._XOA_masks(
                SRT_section
            )
            self.prior_path = self.output_dir / f'XOA_masks_{SRT_section}.npy'
        else:
            self.prior_path = ''

        print(
            f' Priors set to {self.priors}.'
            f'\n Prior_path: {self.prior_path}'
        )

        transcripts = pl.read_parquet(self.SRT_dir / 'transcripts.parquet')
        transcripts = self._filter_for_section(
            transcripts,
            self.SRT_section
        )

        self.transcripts_path = (
            self.output_dir / f'transcripts_{self.SRT_section}.parquet'
        )
        transcripts.write_parquet(
            self.transcripts_path,
            compression='snappy',
            use_pyarrow=True,
            pyarrow_options=dict(
                version = '1.0',
            )
        )

    def get_VSI(self, transcripts, **kwargs):
        cdf = transcripts_for_ovrlpy(
            machine=self.machine,
            file_path=transcripts,
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
            self.SRT_pxsize[0],
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
        counts_path = self.output_dir / 'counts.mtx.gz'
        self.counts = mmread(counts_path, parallelism=mp.cpu_count())

    # def run_leiden():

    # def assign_CT_probabilities():

    # def assign_stain_to_CT():

    # def assign_pixel_to_cell():

    # def {function for running cluster_CellTypes, assign_CT, stain_to_Cell}