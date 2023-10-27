##########
# MODELS #
##########
import os, sys
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, 
                                     MaxPool3D, 
                                     Dropout, 
                                     Input, 
                                     BatchNormalization, 
                                     GlobalAveragePooling3D, 
                                     Dense,
                                     ZeroPadding3D,
                                     Activation,
                                     Add)
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
        if self.model_name == "simple3d":
            return self.__simple3d()
        elif self.model_name == "roimask3d":
            return self.__roimask3d()
        elif self.model_name == "roimask3d1":
            return self.__roimask3d1()
        else:
            raise "Unknow model name."
    
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
        x = Dropout(0.3)(x)

        if self.n_classes == 1:
            outputs = Dense(units=1, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.n_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs, name = "roimask3d")

        return model
    
    
    def __roimask3d1(self):
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
            attentionMap = MaxPool3D(pool_size=2)(attentionMap)
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap) 
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        # conv2
        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        if self.mask_depth > 1:
            attentionMap = self.heart_attention_layer(64)(attentionMap)
            x = x * attentionMap
            attentionMap = MaxPool3D(pool_size=2)(attentionMap)
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        # conv3
        x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        if self.mask_depth > 2:
            attentionMap = self.heart_attention_layer(128)(attentionMap)
            x = x * attentionMap
            attentionMap = MaxPool3D(pool_size=2)(attentionMap)
            if self.is_bn_mask:
                attentionMap = BatchNormalization()(attentionMap)
        x = MaxPool3D(pool_size=2)(x)
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
        x = Dropout(0.3)(x)

        if self.n_classes == 1:
            outputs = Dense(units=1, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.n_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs, name = "roimask3d1")

        return model
    
    
    def __simple3d(self):
        inputs = Input(self.input_size)
        x = Conv3D(filters=self.first_filters, kernel_size=3, activation="relu")(inputs)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        x = MaxPool3D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = GlobalAveragePooling3D()(x)
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.3)(x)
            
        if self.n_classes == 1:
            outputs = Dense(units=1, activation="sigmoid")(x)
        else:
            outputs = Dense(units=self.n_classes, activation="softmax")(x)
            
        model = Model(inputs, outputs, name = "simple3d")
        
        return model
        
        
        
###### 
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
