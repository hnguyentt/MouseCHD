import cv2
import numpy as np
import sys, os, json
import tifffile
from PIL import Image

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model

from skimage.transform import resize
     

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
                loss = preds[0]
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
        cam = np.pad(cam, ((1,1), (3,3), (3,3)),"constant")
        cam = resize(cam, upsample_size)
        cam = (cam - cam.min())/(cam.max()-cam.min())
        
        return cam
    
def overlay3d(image, gradcam):
    cam = gradcam.copy()
    cam = cam / cam.max()
    cam = np.uint8(cam*255)
    overlays = []
    cams = []
    for i in range(image.shape[0]):
        cam3 = np.expand_dims(cam[i,:,:], axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)
        im = image[i,:,:]
        im = (im - im.min())/(im.max()-im.min())
        im = np.uint8(im*255)
        im3 = np.expand_dims(im, axis=2)
        im3 = np.tile(im3, [1, 1, 3])
        overlay = 0.3 * cam3 + 0.5 * im3
        overlay = (overlay*255. / overlay.max()).astype(np.uint8)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        overlays.append(overlay)
        
        cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
        cams.append(cam3)
    
    return np.asarray(cams), np.asarray(overlays)


def generate_parellel(im, overlay, view="axial"):
    views = {"axial": 0, "coronal": 1, "sagittal": 2}
    assert view in views, f"view must be in {views.keys}"
    # im = np.stack([im, im, im], axis=3)
    im = np.uint8((im - im.min())/(im.max() - im.min())*255.)
    parallels = []
    
    for i in range(overlay.shape[views[view]]):
        if view == "axial":
            images = list([Image.fromarray(im[i,:,:]), 
                           Image.fromarray(overlay[i,:,:])])
        elif view == "coronal":
            images = list([Image.fromarray(im[:,i,:]), 
                           Image.fromarray(overlay[:,i,:])])
        else:
            images = list([Image.fromarray(im[:,:,i]), 
                           Image.fromarray(overlay[:,:,i])])

        widths, heights = zip(*(j.size for j in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im1 in images:
            new_im.paste(im1, (x_offset,0))
            x_offset += im1.size[0]
        
        parallels.append(np.array(new_im))
    
    return np.array(parallels)
    


