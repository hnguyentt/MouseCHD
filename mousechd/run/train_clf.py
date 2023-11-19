"""
Train classifier 
"""

import os
import json
import re
import argparse
import pandas as pd
import tensorflow as tf
import logging

from mousechd.classifier.utils import (MODEL_NAMES,
                                       CLF_DIR,
                                       eval_clf,
                                       find_best_ckpt,
                                       calculate_metrics,
                                       download_clf_models)
from mousechd.utils.tools import CACHE_DIR, set_logger
from mousechd.classifier.evaluate import predict_folder, summarize_results
from mousechd.classifier.train import train_clf
from mousechd.classifier.augments import AUGMENT_POLS


def add_args(parser):
    parser.add_argument("-exp_dir", type=str, help="experiment directory")
    parser.add_argument("-exp", type=str, help="name of experiment")
    parser.add_argument("-data_dir", type=str, help="path to data directory")
    parser.add_argument("-label_dir", type=str, help="path to directory containing label files")
    parser.add_argument("-test_path", type=str, help="path to test file", default=None)
    parser.add_argument("-testdir", type=str, help="test datadir", default=None)
    parser.add_argument("-test_bz", type=int, help="test batch size", default=16)
    parser.add_argument("-configs", type=str, help="path to configs file", default=None)
    parser.add_argument("-log_dir", type=str, help="Logging directory for tensorboard",
                        default=os.path.join(CACHE_DIR, "LOGS", "Classifier"))
    parser.add_argument("-evaluate", help="evaluate: best, none, all", type=str,
                        choices=["best", "none", "all"], default="none")
    parser.add_argument("-logfile", type=str, help="path to logfile", default=None)
    parser.add_argument("-epochs", type=int, help="epochs", default=None)
    
MONITOR_LIST = ["val_loss","val_accuracy", "val_weighted_accuracy"]

def main(args):
    download_clf_models()
    # Process arguments
    if args.configs is None:
        configs = json.load(open(os.path.join(CLF_DIR, "configs.json"), "r"))
    else:
        configs = json.load(open(args.configs, "r"))
    if args.epochs is not None:
        configs["epochs"] = args.epochs
        
    if configs["n_classes"] == 1:
        assert configs["loss_fn"] in ["binary_crossentropy","sigmoid_focal_crossentropy"], "loss_fn for n_classes=1: binary_crossentropy,sigmoid_focal_crossentropy"
    else:
        assert configs["loss_fn"] == "categorical_crossentropy", "n_classes={}, only categorical_crossentropy can be applied as loss_fn".format(configs["n_classes"])
    
    assert configs["model_name"] in MODEL_NAMES, f"'model_name' must be in {MODEL_NAMES}"
    assert configs["mask_depth"] in [i for i in range(1,5)], "'mask_depth' must be in {}".format([i for i in range(1,5)])
    assert configs["augment"] in AUGMENT_POLS, f"'augment' must be in {AUGMENT_POLS}"
    assert configs["monitor"] in MONITOR_LIST, f"'monitor' must be in {MONITOR_LIST}"
    
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = os.path.join(args.label_dir, "test.csv")
     
    
    save_dir = os.path.join(args.exp_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)
    if args.logfile is not None:
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        set_logger(args.logfile)
    else:
        set_logger(os.path.join(save_dir, "training.log"))
    
    with open(os.path.join(save_dir, "configs.json"),"w") as f:
        json.dump(configs, f, indent=1)
    
    # TRain  
    logging.info("="*15 + "//" + "="*15)
    logging.info("TRAIN") 
    model = train_clf(save_dir=save_dir,
                      exp=args.exp,
                      data_dir=args.data_dir,
                      label_dir=args.label_dir, 
                      configs=configs,
                      log_dir=args.log_dir)
    
    # Evaluate
    if args.evaluate != "none":
        logging.info("="*15 + "//" + "="*15)
        if args.testdir is None:
            testdir = args.data_dir
        else:
            testdir = args.testdir
        logging.info("EVALUATE")
        sum_df_path = os.path.join(save_dir, "results", "summary.csv")
        try:
            sum_df = pd.read_csv(sum_df_path)
        except FileNotFoundError:
            sum_df = pd.DataFrame(columns=["ckpt", "testfile",
                                        "acc", "bal_acc",
                                        "sens", "spec", "auc"])
        df = pd.read_csv(test_path)
        ## Restore ckpt
        if args.evaluate == "best":
            ckpts = [find_best_ckpt(save_dir, re.sub(r"^val_", "", configs["monitor"]))]
        else:
            ckpts = sorted([x for x in os.listdir(save_dir)
                            if x.endswith(".hdf5") and (not x.startswith("."))])
        
        for ckpt in ckpts:
            logging.info("Restore: {}".format(ckpt))
            model.load_weights(os.path.join(save_dir, ckpt))
            
            res = predict_folder(model=model,
                                imdir=testdir,
                                maskdir=None,
                                target_size=configs["input_size"],
                                label_df=df,
                                stage="eval",
                                batch_size=args.test_bz,
                                save=os.path.join(save_dir, "results", ckpt, os.path.basename(test_path)),
                                grouped_result=True)
            
            sum_df = summarize_results(df=res,
                                    ckpt=ckpt,
                                    test_fn=os.path.basename(test_path),
                                    sum_df=sum_df,
                                    grouped_result=True)
            
            sum_df.to_csv(sum_df_path, index=False)
        