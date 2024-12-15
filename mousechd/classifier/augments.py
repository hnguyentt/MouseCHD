#########################
# AUGMENTATION POLICIES #
#########################
from volumentations import (Compose,
                            Rotate,
                            GaussianNoise,
                            RandomRotate90,
                            RandomBrightnessContrast)

AUGMENT_POLS = [None, "augment0", "augment1"]

def augment0(im, mask=None):
    """
    Augmentation for training data. (policy 0)
    Args:
        im (np.array): image array
    Returns:
        np.array: augmented image array
    """
    aug = Compose([
        Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.5),
    ], p=1.0)
    
    if mask is None:
        return aug(**{'image': im})['image']
    else:
        transformed = aug(**{'image': im, 'mask': mask})

        return transformed
    
    
def augment1(im, mask=None):
    aug = Compose([
        Rotate((-180, 180), (0, 0), (0, 0), p=0.8),
        RandomRotate90((1, 2), p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2,
                                 contrast_limit=0.2,
                                 brightness_by_max=True,
                                 p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.5),
    ], p=1.0)
    
    if mask is None:
        return aug(**{'image': im})['image']
    else:
        transformed = aug(**{'image': im, 'mask': mask})

        return transformed