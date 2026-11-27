import argparse
import gc
import os

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "20"
os.environ["RUST_BACKTRACE"] = "full"

from rich.traceback import install

install(show_locals=True)

from phenium.main import Phenium
from torch.cuda import empty_cache

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="analyse")
    parser.add_argument(
        "-x",
        "--Xenium",
        type=str,
        help="Path to the directory containing the Xenium recordings.",
    )
    parser.add_argument(
        '-p',
        '--PhenoCycler',
        type=str,
        help='Path to the phenocycler image.'
    )
    parser.add_argument(
        "-o",
        "--OutDir",
        type=str,
        help="The output directory to save the results in."
    )
    parser.add_argument(
        "-c",
        "--Config",
        type=str,
        help="Path to the config file."
    )
    parser.add_argument(
        '-pt',
        '--PriorType',
        type=str,
        default=None,
        help='The type of prior to use for ProSeg.', choices=['XOA', 'CP', 'None']
    )
    parser.add_argument(
        '-ch',
        '--Channels',
        nargs="*",
        default=None,
        help='The channels to use for segmenting with CellPose. Either by name or index'
    )
    parser.add_argument(
        '-e',
        '--Exclude',
        nargs="*",
        default=None,
        help='Channels to exclude from any and all steps. Either by name or index.'
    )
    parser.add_argument(
        '-cs',
        '--CellSize',
        type=int,
        default=8,
        help='Size of a cell in micro meter. Consider the swelling caused by hydration.'
    )
    parser.add_argument(
        '-s',
        '--Seed',
        type=int,
        default=None,
        help='Seed of the random-generators.'
    )

    args = parser.parse_args()

    if args.PriorType == 'None':
        args.PriorType = None

    ana = Phenium(
        SRT_dir=args.Xenium,
        IHC_path=args.PhenoCycler,
        output_dir=args.OutDir,
        config_path=args.Config,
        prior_type=args.PriorType,
        channels=args.Channels,
        exclude=args.Exclude,
        cell_size=args.CellSize,
        seed=args.Seed,
    )

    for section in ana.XMI.sections:
        ana.prepare(section, 0, skip_affine=True, verbose=True)
        ana.run_proseg()
        ana._anndata(make_unique=True)
        gc.collect()
        _ = empty_cache()