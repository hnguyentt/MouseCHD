import os
import re, json
import glob
import numpy as np
import pandas as pd
import tifffile
import SimpleITK as sitk
from skimage.transform import resize
import logging 

import tensorflow as tf
from tensorflow.keras.layers import (Conv3D, 
                                     MaxPool3D, 
                                     Dropout, 
                                     Input, 
                                     BatchNormalization, 
                                     GlobalAveragePooling3D, 
                                     Dense)
from tensorflow.keras import Model

from mousechd.utils.tools import set_logger
from mousechd.classifier.gradcam import overlay3d, generate_parellel
from mousechd.datasets.utils import (norm_min_max, 
                                        resample3d, 
                                        get_largest_connectivity, 
                                        crop_heart_bbx,
                                        maskout_non_heart)

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


class GradCAM3D:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if "tf.math.multiply" in layer.name:
                print(layer.name)
                return layer.name
            elif (len(layer.output_shape) == 5) and "conv" in layer.name:
                print(layer.name)
                return layer.name
        raise ValueError("Could not find 5D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)
            if classIdx is None:
                loss = preds[:, np.argmax(preds)]
            else:
                loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1,2))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = np.pad(cam, ((4,4), (4,4), (4,4)),"constant")
        cam = resize(cam, upsample_size)
        cam = (cam - cam.min())/(cam.max()-cam.min())
        
        return cam, preds


uncertain_cases = {
    "F1": ["RP5D1"],
    "F3": ["NH_1045"],
    "F4": ["NH_344", "N_255h"]
}

HOME = os.environ.get("HOME")
exp_dir = f"{HOME}/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/Classifier"
outdir = f"{HOME}/DATA/INCEPTION_2020-CHD/Mice/OUTPUTS/GradCAMs_uncertainty1"
imdir = f"{HOME}/DATA/INCEPTION_2020-CHD/Mice/DATA/CTs/resampled/Imagine/images"


def explain_one(grad_model, input_shape, imls, bootstrap_idx):
    
    csv_records = []
    for i, imname in enumerate(imls):
        logging.info("{}. {}".format(i+1, imname))
        img = sitk.ReadImage(os.path.join(imdir, imname))
        
        ori_im = sitk.GetArrayFromImage(img)
        img = resample3d(img, input_shape[::-1])
        im = sitk.GetArrayFromImage(img)
        im = norm_min_max(im)
        im = np.expand_dims(im, axis=3)
        im = np.expand_dims(im, axis=0)
        
        cam, preds = grad_model.compute_heatmap(image=im,
                                                classIdx=None,
                                                upsample_size=ori_im.shape)
        colored_cam, colored_overlay = overlay3d(ori_im, cam)
        
        class_idx = np.argmax(preds)
        logging.info(preds[0, 1].numpy())
        
        savedir = os.path.join(outdir, f"bootstrap{bootstrap_idx:02d}", imname)
        
        os.makedirs(savedir, exist_ok=True)
        
        # save cam
        tifffile.imwrite(os.path.join(savedir, f"cam_{class_idx}.tif"), colored_cam)
        # save overlay cam
        tifffile.imwrite(os.path.join(savedir, f"overlay_{class_idx}.tif"), colored_overlay)
        # Save im as tif
        tifffile.imwrite(os.path.join(savedir, "image.tif"), np.uint8(ori_im*255))
        
        # Parallel
        axial_parallel = generate_parellel(ori_im, colored_overlay, view="axial")
        tifffile.imwrite(os.path.join(savedir, f"parallel_axial_{class_idx}.tif"), axial_parallel)
        coronal_parallel = generate_parellel(ori_im, colored_overlay, view="coronal")
        tifffile.imwrite(os.path.join(savedir, f"parallel_coronal_{class_idx}.tif"), coronal_parallel)
        sagittal_parallel = generate_parellel(ori_im, colored_overlay, view="sagittal")
        tifffile.imwrite(os.path.join(savedir, f"parallel_sagittal_{class_idx}.tif"), sagittal_parallel)
        
        csv_records.append([f"bootstrap{bootstrap_idx:02d}", imname, preds[0, 1].numpy(), class_idx])
        
    return csv_records


def main():
    
    os.makedirs(outdir, exist_ok=True)
    
    set_logger(os.path.join(outdir, "explain.log"))
    
    csv_records = []
    
    for fold, imls in uncertain_cases.items():
        # Load model
        logging.info(fold)
        model = load_MouseCHD_model(conf_path=os.path.join(exp_dir, fold, "configs.json"),
                                    weights_path=os.path.join(exp_dir, fold, "best_model.hdf5"))
        input_shape = model.layers[0].output_shape[0][1:4]
        
        grad_model = GradCAM3D(model, layerName=None)
        
        logging.info("Total: {}".format(len(imls)))
        
        for i in range(1, 51):
            logging.info(f"bootstrap{i:02d}")
            csv_records += explain_one(grad_model, input_shape, imls, i)
        
    df = pd.DataFrame(csv_records, columns=["bootstrap", "heart_name", "prob", "pred"])
    df.to_csv(f"{outdir}/probabilities.csv", index=False)
        
        
main()    

