"""
Train models 
"""
import os, re
import json
import pandas as pd
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
                                        ModelCheckpoint,
                                        EarlyStopping,
                                        TensorBoard,
                                        CSVLogger)

from sklearn.utils import class_weight

from .datagens import MouseCHDGen
from .utils import load_label, calculate_metrics, download_clf_models, CLF_DIR
from .models import load_MouseCHD_model, MouseCHD
from .evaluate import predict_folder


def train_clf(save_dir,
              exp,
              data_dir, 
              label_dir, 
              configs,
              log_dir):     
    train_df = load_label(os.path.join(label_dir, "train.csv"), configs["seed"])
    val_df = load_label(os.path.join(label_dir,"val.csv"), configs["seed"])
    
    logging.info("TRAIN:")
    logging.info(train_df["label"].value_counts())
    logging.info("VAL:")
    logging.info(val_df["label"].value_counts())
    
    
    # optimizer
    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=configs["lr"],
                                                                  decay_steps=1000,
                                                                  decay_rate=configs["decay"])
    optimizer = tf.keras.optimizers.SGD(learning_rate = lr_scheduler, 
                                        momentum = configs["momentum"], 
                                        nesterov = True)
    
    # loss_fn
    if configs["loss_fn"] == "categorical_crossentropy":
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # elif configs["loss_fn"] == "sigmoid_focal_crossentropy":
    #     loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=configs["alpha"], gamma=configs["gamma"])
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        
    # class weights
    if configs["class_weights"]:
        weights = class_weight.compute_class_weight(class_weight="balanced", 
                                                    classes=np.unique(train_df["label"]), 
                                                    y=train_df["label"].values)
        weights = {i: weights[i] for i in range(len(train_df["label"].unique()))}
    else:
        weights = None
        
    logging.info("Class weights: {}".format(weights))
    
    # model
    model = MouseCHD(model_name=configs["model_name"],
                     input_size=configs["input_size"],
                     n_classes=configs["n_classes"],
                     first_filters=configs["first_filters"],
                     mask_depth=configs["mask_depth"],
                     is_bn_mask=configs["is_bn_mask"]).build_model()
        
    # Metrics
    metrics = ["accuracy", 
               tf.keras.metrics.Recall(), 
               tf.keras.metrics.Precision()]
    
    # Callbacks
    if configs["save_best"]:
        save_model_name = "best_model.hdf5"
    else:
        save_model_name = "epoch-{epoch:03d}.hdf5"
    checkpoint = ModelCheckpoint(os.path.join(save_dir, save_model_name),
                                 monitor=configs["monitor"],
                                 verbose=1,
                                 save_best_only=configs["save_best"],
                                 save_weights_only=True,
                                 save_freq="epoch")
    early_stop = EarlyStopping(configs["monitor"], patience=configs["patience"])
    tb = TensorBoard(log_dir=os.path.join(log_dir, exp),
                     update_freq="batch")
    csv_logger = CSVLogger(os.path.join(save_dir, "train.csv"), append=True)
    
    # Data generators
    train_gen = MouseCHDGen(imdir=data_dir,
                            filenames=train_df["heart_name"].values,
                            batch_size=configs["batch_size"],
                            target_size=configs["input_size"],
                            labels=train_df["label"].values,
                            seed=configs["seed"],
                            n_classes=configs["n_classes"],
                            stage="train",
                            augment=configs["augment"],
                            class_weights=weights)
    val_gen = MouseCHDGen(imdir=data_dir,
                          filenames=val_df["heart_name"].values,
                          batch_size=configs["batch_size"],
                          labels=val_df["label"].values,
                          seed=configs["seed"],
                          target_size=configs["input_size"],
                          n_classes=configs["n_classes"],
                          stage="val",
                          augment=None,
                          class_weights=weights) 
    
    # Evaluate initial model and resume   
    if configs["resume"] is not None:
        try:
            model.load_weights(os.path.join(save_dir, configs["resume"]))
        except FileNotFoundError:
            logging.info("Resumed weights not found, retrain from default weights")
            download_clf_models()
            model.load_weights(os.path.join(CLF_DIR, "best_model.hdf5"))
            
        try:
            initial_epoch = int(configs["resume"].split("-")[1])
        except IndexError:
            initial_epoch = 0
    else:
        initial_epoch = 0
        model.save_weights(os.path.join(save_dir, "initial_weights.hdf5"))
        if not os.path.isfile(os.path.join(save_dir, "train.csv")):
            logging.info("="*15 + "//" + "="*15)
            logging.info("Evaluate initial model:")
            logging.info("On training: ")
            train_res = predict_folder(model=model,
                                       imdir=data_dir,
                                       maskdir=None,
                                       target_size=configs["input_size"],
                                       label_df=train_df,
                                       stage="eval",
                                       batch_size=configs["batch_size"],
                                       save=None,
                                       grouped_result=False)
            val_res = predict_folder(model=model,
                                     imdir=data_dir,
                                     maskdir=None,
                                     target_size=configs["input_size"],
                                     label_df=val_df,
                                     stage="eval",
                                     batch_size=configs["batch_size"],
                                     save=None,
                                     grouped_result=False)
            df =  pd.DataFrame(columns=["epoch", 
                                        "loss", 
                                        "accuracy", 
                                        "recall", 
                                        "precision", 
                                        "weighted_accuracy",
                                        "val_loss", 
                                        "val_accuracy", 
                                        "val_recall", 
                                        "val_precision", 
                                        "val_weighted_accuracy"])
            train_loss = tf.keras.metrics.binary_crossentropy(tf.constant(train_res["label"].values.astype(float)),
                                                              tf.constant(train_res["prob"].values.astype(float)))
            val_loss = tf.keras.metrics.binary_crossentropy(tf.constant(val_res["label"].values.astype(float)),
                                                            tf.constant(val_res["prob"].values.astype(float)))
            train_metrics = calculate_metrics(train_res["prob"].values.astype(float),
                                              train_res["label"].values.astype(float))
            val_metrics = calculate_metrics(val_res["prob"].values.astype(float),
                                            val_res["label"].values.astype(float))
            df.loc[len(df), :] = [0, 
                                  train_loss.numpy(), train_metrics["acc"], train_metrics["sens"], train_metrics["spec"], train_metrics["bal_acc"],
                                  val_loss.numpy(), val_metrics["acc"], val_metrics["sens"], val_metrics["spec"], val_metrics["bal_acc"]]
            
            df.to_csv(os.path.join(save_dir, "train.csv"), index=0)
    
    if weights is not None:
        model.compile(loss=loss_fn, 
                      optimizer=optimizer,
                      metrics=metrics,
                      weighted_metrics=["accuracy"])
    else:
        model.compile(loss=loss_fn,
                      optimizer=optimizer,
                      metrics=metrics)
        
    # Training    
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=configs["epochs"],
                        verbose=1,
                        callbacks=[checkpoint, early_stop, tb, csv_logger],
                        initial_epoch=initial_epoch)
    
    return model    

