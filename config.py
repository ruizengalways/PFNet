"""
PFNet for homography estimation
``Rethinking Planar Homography Estimation Using Perspective Field'' implementation.

Licensed under the Apache License 2.0
Written by Rui Zeng

If you find PFNet useful in your research, please consider citing:

@inproceedings{zeng18rethinking,
  author    = {Rui Zeng and Simon Denman and Sridha Sridharan and Clinton Fookes},
  title     = {Rethinking Planar Homography Estimation Using Perspective Field},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2018},
}

Acknowledgement: The implementation of this repo heavily used codes from MaskRCNN repo: https://github.com/matterport/Mask_RCNN
"""


import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 200

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 400
    IMAGE_MAX_DIM = 4096 * 2
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = False  # currently, the False option is not supported
    PATCH_SIZE = 128
    MARGINAL_PIXEL = 32

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL = np.array([117.3])

    # Learning rate and momentum
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    # Weight decay regularization
    WEIGHT_DECAY = 0.001



    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
