from phenium.utils import (
    to_cli
)

import multiprocessing as mp
from subprocess import Popen

from cellpose import models


def cellpose_wrapper(model_params, eval_params, style):
    model = models.CellposeModel(**model_params)
    if style == 'pheno':
        # load_as_zarr phenocycler
        pass
    elif style == 'focus':
        # load_as_zarr morphology_focus
        pass
    else:
        raise NotImplementedError(f'Style `{style}` is not implemented.')
    
    masks, flows, styles = model.eval(img, **eval_params)


def proseg_wrapper(
    proseg_params,
    transcripts_path,
    output_dir,
    prior_path
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
                        value = self.config['ImageStats']['pixelsize_xy']
                elif arg == 'burnin-voxel-size':
                    if not value:
                        value = self.config['ImageStats']['pixelsize_xy']*8
                else:
                    value = value
                general.append(
                    f'--{arg} {value}'
                )
        elif category == 'output':
            output = list(map(
                to_cli,
                dic.items()
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
        arguments += ['--cellpose-masks', f'{prior_path}']

    arguments.append(transcripts_path)

    cmd = ' '.join(arguments)
    p = Popen(cmd, shell=True)
    p.wait()

    return None