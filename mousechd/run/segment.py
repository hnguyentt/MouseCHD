import os
import argparse

from mousechd.segmentation.segment import segment_from_folder


def add_args(parser):
    parser.add_argument("-indir", type=str, help="Input directory")
    parser.add_argument("-outdir", type=str, help="Output directory")
    parser.add_argument("-fold", type=int, help="Fold to predict. Default: None, meaning ensemble all folds", default=None)
    parser.add_argument("-step_size", type=float, 
                        help=("how much the sliding window moves during the patch-based processing of images."
                              + "The smaller step size results in more overlap between patches, leading to potentially better accuracy but slower inference time!"),
                        default=0.5)
    parser.add_argument("-disable_mixed_precision", type=bool,
                        help="Disable mixed precision", default=False)
    parser.add_argument("-num_threads_preprocessing", type=int,
                        help="Number of threads for preprocessing",
                        default=6)
    parser.add_argument("-num_threads_nifti_save", type=int,
                        help="Number of threads to save NIFTI file",
                        default=2)
    
    return parser


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    segment_from_folder(args.indir, args.outdir, folds=args.fold,
                        step_size=args.step_size,
                        disable_mixed_precision=args.disable_mixed_precision,
                        num_threads_preprocessing=args.num_threads_preprocessing,
                        num_threads_nifti_save=args.num_threads_nifti_save)
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Heart segmentation")
    parser = add_args(parser)
    args = parser.parse_args()
    main()