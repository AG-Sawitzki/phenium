from phenium.utils import (
    to_cli
)

import multiprocessing as mp
from functools import partial
from subprocess import Popen

from cellpose import models


def cellpose_wrapper(img, model_params, eval_params):
    model = models.CellposeModel(**model_params)
    # if style == 'pheno':
    #     # load_as_zarr phenocycler for coords
    #     pass
    # elif style == 'focus':
    #     # load_as_zarr morphology_focus for coords
    #     pass
    # else:
    #     raise NotImplementedError(f'Style `{style}` is not implemented.')
    
    masks, flows, styles = model.eval(img, **eval_params)
    return masks, flows, styles


def proseg_wrapper(
    proseg_params,
    transcripts_path,
    output_dir,
    pixelsize,
    machine,
    prior_path = '',
):
    for category, dic in proseg_params.items():
        if category == 'general':
            general = ['--overwrite']
            for arg, value in dic.items():
                if arg == 'nthreads':
                    if not value:
                        value = mp.cpu_count()
                elif arg == 'voxel-layers':
                    if not value:
                        value = 4
                elif arg == 'voxel-size':
                    if not value:
                        value = pixelsize
                elif arg == 'burnin-voxel-size':
                    if not value:
                        value = pixelsize*8
                else:
                    value = value
                general.append(
                    f'--{arg} {value}'
                )
        elif category == 'output':
            for arg, value in dic.items():
                if value:
                    ext = '.parquet'
                    if arg == 'counts':
                        ext = '.mtx.gz'
                    elif 'cell-polygon' in arg:
                        ext = '.geojson.gz'
                    dic[arg] = f'{arg}{ext}'
            output = list(map(
                partial(
                    to_cli,
                    prefix='output-',
                ),
                dic.items(),
            ))
            output.append(f'--output-path {output_dir}')
        elif category == 'model':
            model = list(map(
                to_cli,
                dic.items()
            ))
        elif category == 'diffusion':
            if not dic['no-diffusion']:
                diffusion = list(map(
                    to_cli,
                    dic.items()
                ))
        else:
            raise KeyError(f'Unknown Category: `{category}`.')

    arguments = general + output + model + diffusion
    if prior_path:
        arguments += ['--cellpose-masks', str(prior_path)]

    arguments.append(f'--{machine}')
    arguments.append(str(transcripts_path))

    arguments.insert(0, 'proseg')

    cmd = ' '.join(arguments)
    print(cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    return None