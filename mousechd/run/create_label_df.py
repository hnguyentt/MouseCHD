import os
import pandas as pd
import argparse

from mousechd.datasets.preprocess import x5_df


def add_args(parser):
    parser.add_argument("-metafile", type=str, help="path to metafile")
    parser.add_argument("-outdir", type=str, help="output directory")
    parser.add_argument("-basename", type=str, help="file basename")
    


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    
    df = pd.read_csv(args.metafile)
    
    if "label" not in df.columns:
        df["label"] = 1 - df["Normal heart"]
        
    df = df[["heart_name", "label"]]
    df_x5 = x5_df(df)
    
    df.to_csv(os.path.join(args.outdir, f"{args.basename}.csv"), index=False)
    df_x5.to_csv(os.path.join(args.outdir, f"{args.basename}_x5.csv"), index=False)