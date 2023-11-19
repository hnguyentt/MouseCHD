import os, re
import argparse
import pandas as pd

from mousechd.classifier.models import load_MouseCHD_model
from mousechd.classifier.evaluate import predict_folder, summarize_results
from mousechd.utils.tools import set_logger
from mousechd.classifier.utils import (download_clf_models, 
                                       CLF_DIR, 
                                       calculate_metrics)


def add_args(parser):
    parser.add_argument("-model_dir", help="model directory", default=None)
    parser.add_argument("-fold", help="fold", choices=["F1", "F2", "F3", "F4", "F5"], default=None)
    parser.add_argument("-ckpt", help="name of checkpoint to restore", default="best_model.hdf5")
    parser.add_argument("-imdir", help="image directory")
    parser.add_argument("-maskdir", help="mask directory", default=None)
    parser.add_argument("-label", help="path to label csv file", default=None)
    parser.add_argument("-stage", help="stage: ['eval', 'test']", choices=["test", "eval"], default="test")
    parser.add_argument("-batch_size", type=int, help="batch size", default=8)
    parser.add_argument("-outdir", help="output directory", default=None)
    parser.add_argument("-grouped_result", help="grouping result?", type=int, choices=[0,1], default=0)
    parser.add_argument("-savename", help="save name for evaluation results", default=None)
    parser.add_argument("-device", help="device: 'cpu' or 'gpu'", choices=['cpu', 'gpu'], default='gpu')
    
    
def main(args):
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.outdir is None:
        if args.fold is None:
            sum_df_path = os.path.join(args.model_dir, "results", "summary.csv")
        else:
            sum_df_path = os.path.join(args.model_dir, args.fold, "results", "summary.csv")
    else:
        sum_df_path = os.path.join(args.outdir, "summary.csv")
        print("DEBUG: {}".format(sum_df_path))
        
    if args.savename is not None:
        savename = args.savename
    else:
        if args.label is None:
            savename = "test.csv"
        else:
            savename = os.path.basename(args.label)
    
    # Summary dataframe
    try:
        sum_df = pd.read_csv(sum_df_path)
    except FileNotFoundError:
        sum_df = pd.DataFrame(columns=["ckpt", "testfile",
                                       "acc", "bal_acc",
                                       "sens", "spec", "auc"])
    
    # Download models if necessary
    if args.model_dir is None:
        download_clf_models()
        conf_path = os.path.join(CLF_DIR, "configs.json")
    else:
        if args.fold is None:
            conf_path = os.path.join(args.model_dir, "configs.json")
        else:
            conf_path = os.path.join(args.model_dir, args.fold, "configs.json")
    
    # List of checkpoints needing evaluation    
    if args.ckpt == "all":
        ckpts = sorted([x for x in os.listdir(os.path.dirname(conf_path))
                        if x.endswith(".hdf5") and (not x.startswith("."))])
    else:
        ckpts = [args.ckpt]
    
    # Start evaluating    
    for ckpt in ckpts:
        if args.outdir is None:
            outdir = os.path.join(os.path.dirname(sum_df_path), ckpt)
        else:
            outdir = args.outdir
            
        os.makedirs(outdir, exist_ok=True)
        set_logger(os.path.join(outdir, re.sub(r".csv$", ".log", savename)))
        
        model = load_MouseCHD_model(conf_path=conf_path,
                                    weights_path=os.path.join(os.path.dirname(conf_path), ckpt))
        target_size = model.layers[0].output_shape[0][1:]
        
        label_df = pd.read_csv(args.label)
        
        df = predict_folder(model=model,
                            imdir=args.imdir,
                            maskdir=args.maskdir,
                            target_size=target_size,
                            label_df=label_df,
                            stage=args.stage,
                            batch_size=args.batch_size,
                            save=os.path.join(outdir, savename),
                            grouped_result=bool(args.grouped_result))
        
        # Save summary results
        sum_df = summarize_results(df=df,
                                   ckpt=ckpt,
                                   test_fn=savename,
                                   sum_df=sum_df,
                                   grouped_result=args.grouped_result)
        
        sum_df.to_csv(sum_df_path, index=False)
    
    