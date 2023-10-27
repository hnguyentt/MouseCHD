"""
Splitting data for training, testing and validation 
"""
from ast import arg
import logging
import os, sys
import argparse
import pandas as pd

from mousechd.datasets.preprocess import (split_data,
                                          split_k_folds)
from mousechd.utils.tools import set_logger

def add_args(parser):
    parser.add_argument("-metafile", type=str, help="path to the meta file")
    parser.add_argument("-sep", type=str, help="separator in the meta file", default=",")
    parser.add_argument("-outdir", type=str, help="output directory")
    parser.add_argument("-num_folds", type=int, help="number of folds", default=1)
    parser.add_argument("-val_size", type=float, help="validation size", default=0.1)
    parser.add_argument("-seed", type=int, help="seed for randomization", default=42)

    return parser

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    set_logger(os.path.join(args.outdir, "split_data_{}.log".format(args.num_folds)))
    
    logging.info("-metafile: {}".format(args.metafile))
    logging.info("-sep: {}".format(args.sep))
    logging.info("-outdir: {}".format(args.outdir))
    logging.info("-num_folds: {}".format(args.num_folds))
    logging.info("-val_size: {}".format(args.val_size))
    logging.info("-seed: {}".format(args.seed))
    
    meta_df = pd.read_csv(args.metafile, sep=args.sep)
    
    if args.num_folds > 1:
        split_k_folds(meta_df=meta_df,
                      num_folds=args.num_folds,
                      val_size=args.val_size,
                      outdir=args.outdir,
                      seed=args.seed)
    else:
        split_data(meta_df=meta_df,
                   outdir=args.outdir,
                   val_size=args.val_size,
                   seed=args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SPLITTING DATA FOR TRAINING, TESTING AND VALIDATION")
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    