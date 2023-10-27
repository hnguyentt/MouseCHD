###################
# DATA GENERATORS # 
###################

import os, sys
import numpy as np
import cv2 
from skimage.transform import resize
import SimpleITK as sitk 
import tensorflow as tf
import logging

from ..datasets.utils import (resample3d, 
                              get_largest_connectivity,
                              crop_heart_bbx,
                              maskout_non_heart,
                              norm_min_max,
                              make_isotropic,
                              pad_image)
import mousechd.classifier.augments as augments


class MouseCHDGen(tf.keras.utils.Sequence):
    def __init__(self,
                 imdir,
                 filenames,
                 batch_size,
                 target_size,
                 labels=None,
                 seed=42,
                 n_classes=1,
                 stage="train",
                 augment=None,
                 class_weights=None,
                 **kwargs):
        """Initialization

        Args:
            imdir (str): path to image directory
            filenames (ls): list of image files
            labels (ls): corresponding labels
            batch_size (int): batch size
            target_size (tuple): target size
            seed (int, optional): random seed. Defaults to 42.
            n_classes (int, optional): number of classes. Defaults to 1.
            stage (str, optional): 'train', 'test', 'val', or 'eval'. Defaults to "train".
            class_weights (tuple, optional): class weights. Defaults to None.
        """
        
        self.imdir = imdir
        self.filenames = filenames
        self.n_classes = n_classes
        if labels is None:
            labels = [0] * len(filenames)
            
        self.numeric_labels = labels
        if n_classes == 1:
            self.labels = np.array(labels)
        else:
            self.labels = tf.keras.utils.to_categorical(np.array(labels), num_classes=n_classes)
            
        self.list_IDs = [i for i in range(len(filenames))]
        self.batch_size = batch_size
        self.target_size = target_size
        self.seed = seed
        self.stage = stage
        self.augment = augment
        self.class_weights = class_weights
        self.kwargs = kwargs
        self.on_epoch_end()
        
    
    def __len__(self):
        return int(np.ceil(len(self.filenames)/float(self.batch_size)))
    

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        return self.__datagen(list_ids_tmp)
        
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.stage in ["train", "val"]:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
            
    
    def __datagen(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = np.empty((self.batch_size, self.n_classes))
        
        if self.class_weights is not None:
            weights = np.empty((self.batch_size, 1))
            
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            img = sitk.ReadImage(os.path.join(self.imdir, self.filenames[ID]))
            img = resample3d(img, self.target_size[:3][::-1])
            im = sitk.GetArrayFromImage(img)
            im = norm_min_max(im)
            
            if self.augment is not None:
                getattr(augments, self.augment)(im, mask=(im != 0))
                
            if self.target_size[3] > 1:
                im = np.stack([im]*self.target_size[3], axis=3)
            else:
                im = np.expand_dims(im, axis=3)
                
            X[i,:] = im
            y[i,:] = self.labels[ID]
            
            files.append(self.filenames[ID])
            
            if self.class_weights is not None:
                weights[i,:] = self.class_weights[self.numeric_labels[ID]]
                
        
        if self.stage in ["test", "eval"]:
            return X, y, files
        else:
            if self.class_weights is None:
                return X, y
            else:
                return X, y, weights
                
                
class MouseCHDEvalGen(tf.keras.utils.Sequence):
    def __init__(self,
                 imdir,
                 target_size,
                 maskdir=None,
                 filenames=None,
                 batch_size=8,
                 labels=None,
                 stage="eval"):
        
        assert stage in ["eval", "test"], "stage must be in ['eval', 'test']"
        
        self.imdir = imdir
        self.maskdir = maskdir
        if filenames is None:
            if maskdir is None:
                filenames = [re.sub(r".nii.gz$", "", x) for x in os.listdir(imdir)]
            else:
                filenames = [re.sub(r".nii.gz$", "", x) for x in os.listdir(maskdir)]
        self.filenames = filenames
        self.batch_size = batch_size
        self.target_size = target_size
        self.list_IDs = [i for i in range(len(filenames))]
        
        if labels is None:
            labels = [None] * len(filenames)
        self.labels = labels
        
        self.stage = stage
        
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.filenames)/float(self.batch_size)))
    

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_tmp = [self.list_IDs[k] for k in indexes]
        
        if self.maskdir is None:
            return self.__datagen_from_resampled(list_ids_tmp)
        else:
            return self.__datagen(list_ids_tmp)
        
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
            
    
    def __datagen(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = []
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            img = sitk.ReadImage(os.path.join(self.imdir, "{}_0000".format(self.filenames[ID])))
            im = sitk.GetArrayFromImage(img)
            
            mask = sitk.ReadImage(os.path.join(self.maskdir, self.filenames[ID]))
            ma = sitk.GetArrayFromImage(mask)
            max_clump = get_largest_connectivity(ma)
            
            cropped_im, cropped_ma = crop_heart_bbx(im, max_clump, pad=(5,5,5))
            resampled_im = maskout_non_heart(cropped_im, cropped_ma)
            resampled_im = norm_min_max(resampled_im)
            
            resampled_img = sitk.GetImageFromArray(resampled_im)
            resampled_img.SetSpacing(img.GetSpacing())
            img = resample3d(resampled_img, self.target_size[:3][::-1])
            im = sitk.GetArrayFromImage(img)
            im = norm_min_max(im)
            im = np.expand_dims(im, axis=3)
            
            X[i,:] = im
            y.append(self.labels[ID])
            files.append(self.filenames[ID])
            
        return X, y, files
        
    
    def __datagen_from_resampled(self, list_ids_tmp):
        X = np.empty((self.batch_size, *self.target_size))
        y = []
        files = []
        
        for i, ID in enumerate(list_ids_tmp):
            img = sitk.ReadImage(os.path.join(self.imdir, self.filenames[ID]))
            img = resample3d(img, self.target_size[:3][::-1])
            im = sitk.GetArrayFromImage(img)
            im = norm_min_max(im)
            im = np.expand_dims(im, axis=3)
            
            X[i,:] = im
            y.append(self.labels[ID])
            files.append(self.filenames[ID])
            
        return X, y, files
        
                        
            