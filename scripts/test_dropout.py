import os, re, json
import argparse
import pandas as pd
import logging
from datetime import datetime

from mousechd.classifier.utils import (download_clf_models, 
                                       CLF_DIR)

import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, 
                                     MaxPool3D, 
                                     Dropout, 
                                     Input, 
                                     BatchNormalization, 
                                     GlobalAveragePooling3D, 
                                     Dense)
from tensorflow.keras import Model

class MouseCHD():
    def __init__(self, 
                 model_name, 
                 input_size, 
                 n_classes, 
                 first_filters=64,
                 mask_depth=4, 
                 is_bn_mask=True):
        self.input_size = input_size
        self.n_classes = n_classes
        self.model_name = model_name
        self.first_filters = first_filters
        self.mask_depth = mask_depth
        self.is_bn_mask = is_bn_mask
        self.model = self.build_model()
        
    def build_model(self):
        return self.__roimask3d()

    
    @staticmethod
    def heart_attention_layer(nb_filters):
        mask_conv = Conv3D(filters=nb_filters, 
                           kernel_size=3, 
                           activation="relu",
                           kernel_initializer = tf.keras.initializers.Ones(),
                           bias_initializer = tf.keras.initializers.Zeros())
        mask_conv.trainable = False
        return mask_conv
    
    
    def __roimask3d(self):
        inputs = Input(self.input_size)
        masks = tf.zeros(self.input_size)
        
        if self.mask_depth > 0:
            masks = tf.less(masks, inputs)
            masks = tf.cast(masks, tf.float32)
        # conv1
        x = Conv3D(filters=self.first_filters, kernel_size=3, activation="relu")(inputs)
        if self.mask_depth > 0:
            attentionMap = self.heart_attention_layer(self.first_filters)(masks)
            x = x * attentionMap
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap) 
        x = BatchNormalization()(x)
        
        # conv2
        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        if self.mask_depth > 1:
            attentionMap = self.heart_attention_layer(64)(attentionMap)
            x = x * attentionMap
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap)
        x = BatchNormalization()(x)
        
        # conv3
        x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        if self.mask_depth > 2:
            attentionMap = self.heart_attention_layer(128)(attentionMap)
            x = x * attentionMap
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap)
        x = BatchNormalization()(x)
        
        # conv4
        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        if self.mask_depth > 3:
            attentionMap = self.heart_attention_layer(256)(attentionMap)
            x = x * attentionMap
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x, training=True) # Set dropout on evaluation as well

        if self.n_classes == 1:
            outputs = Dense(units=1, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.n_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs, name = "roimask3d")

        return model
    

def load_MouseCHD_model(conf_path, weights_path=None):
    """Load MouseCHD model

    Args:
        conf_path (str): path to config.json file
        weights_path (str, optional): path to weights file. Defaults to None.

    Returns:
        keras.Model: MouseCHD model
    """
    model_config = json.load(open(conf_path, "r"))

    model = MouseCHD(model_config["model_name"],
                     model_config["input_size"],
                     model_config["n_classes"],
                     model_config["first_filters"],
                     model_config["mask_depth"],
                     model_config["is_bn_mask"]).build_model()
    
    if weights_path is not None:
        model.load_weights(weights_path)

    return model
    

parser = argparse.ArgumentParser()
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


def set_logger(log_path):
    from imp import reload
    reload(logging)
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
    handlers=[logging.FileHandler(log_path),
              logging.StreamHandler()],
    format='%(message)s',
    level=logging.INFO)
    
    # Start every log with the date and time
    logging.info("="*15 + "//" + "="*15)
    logging.info(datetime.now())


def main(args):
    from mousechd.classifier.evaluate import predict_folder, summarize_results

    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if args.outdir is None:
        if args.fold is None:
            sum_df_path = os.path.join(args.model_dir, "results", "summary.csv")
        else:
            sum_df_path = os.path.join(args.model_dir, args.fold, "results", "summary.csv")
    else:
        sum_df_path = os.path.join(args.outdir, "summary.csv")
        
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
        
        
args = parser.parse_args()
main(args)

""" 
WORKDIR="$HOME/DATA/INCEPTION_2020-CHD/Mice"
./DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/scripts/11_test_dropout.sh F1
"""