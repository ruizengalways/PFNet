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


import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL

import keras.models as KM
import cv2


import utils




# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
#  Utility Functions
############################################################
def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typicallly: [N, 4], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def mace(y_true, y_pred):
    """
    This function is not the final evaluation metric because this function does not use RANSAC to filter out outliers
        for performance improvement.

    This function is only used to indicate a approximate result in training phase.
    :param y_true:
    :param y_pred:
    :return:
    """
    temp = (y_true - y_pred)

    delta_branch1x = tf.gather(temp, 0, axis=3)
    delta_branch1y = tf.gather(temp, 1, axis=3)

    delta_x = delta_branch1x
    delta_y = delta_branch1y

    delta_x_shape = K.shape(delta_x)
    delta_y_shape = K.shape(delta_y)
    delta_x_vec = tf.reshape(delta_x, (delta_x_shape[0], -1))
    delta_y_vec = tf.reshape(delta_y, (delta_y_shape[0], -1))

    delta_x_vec_left_top_corner = tf.gather(delta_x_vec, 0, axis=1)
    delta_x_vec_right_top_corner = tf.gather(delta_x_vec, 127, axis=1)
    delta_x_vec_right_bottm_corner = tf.gather(delta_x_vec, 16383, axis=1)
    delta_x_vec_left_bottm_corner = tf.gather(delta_x_vec, 16256, axis=1)

    delta_y_vec_left_top_corner = tf.gather(delta_y_vec, 0, axis=1)
    delta_y_vec_right_top_corner = tf.gather(delta_y_vec, 127, axis=1)
    delta_y_vec_right_bottm_corner = tf.gather(delta_y_vec, 16383, axis=1)
    delta_y_vec_left_bottm_corner = tf.gather(delta_y_vec, 16256, axis=1)

    dis_x_left_top_corner = K.sqrt(
        K.square(delta_x_vec_left_top_corner) + K.square(delta_y_vec_left_top_corner))
    dis_x_right_top_corner = K.sqrt(
        K.square(delta_x_vec_right_top_corner) + K.square(delta_y_vec_right_top_corner))
    dis_x_right_bottm_corner = K.sqrt(
        K.square(delta_x_vec_right_bottm_corner) + K.square(delta_y_vec_right_bottm_corner))
    dis_x_left_bottm_corner = K.sqrt(
        K.square(delta_x_vec_left_bottm_corner) + K.square(delta_y_vec_left_bottm_corner))

    _mace = K.mean((
                       dis_x_left_top_corner + dis_x_right_top_corner + dis_x_right_bottm_corner + dis_x_left_bottm_corner) / 4)

    return _mace

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, branch, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(branch) + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(branch) + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(branch) + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, branch, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(branch) + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(branch) + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(branch) + str(stage) + block + '_out')(x)
    return x

def deconv_block(input_tensor, kernel_size, filters, stage, branch, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(branch) + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(branch) + str(stage) + block + '_branch'

    x = KL.Conv2DTranspose(nb_filter1, (2, 2), strides=strides, padding='same',
                  name=conv_name_base + '2a', use_bias=False)(input_tensor)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)

    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = KL.Conv2DTranspose(nb_filter3, (2, 2), strides=strides,
                                  name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(branch) + str(stage) + block + '_out')(x)
    return x



def pfnet_graph(input_image, architecture, branch, stage5=False):
    assert architecture in ["resnet30", "resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='res' + str(branch) + '_conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn' + str(branch) + '_bn_conv1')(x)
    x = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, branch=branch, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, branch=branch, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, branch=branch, block='c')
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, branch=branch, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, branch=branch, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3,branch=branch, block='c')
    x = identity_block(x, 3, [128, 128, 512], branch=branch, stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, branch=branch, block='a')
    block_count = {"resnet30": 2, "resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], branch=branch, stage=4, block=chr(98 + i))

    # # Deconv stage 1 /stage 5
    x = deconv_block(x, 3, [1024, 1024, 512], stage=5, branch=branch, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=5, branch=branch, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=5, branch=branch, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=5, branch=branch,  block='d')
    # Deconv stage 2 /stage 6
    x = deconv_block(x, 3, [512, 512, 256], stage=6, branch=branch, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=6, branch=branch, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=6, branch=branch,  block='c')

    # Deconv stage 3/stage 7
    x = deconv_block(x, 3, [256, 256, 128], stage=7, branch=branch, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=7, branch=branch, block='b')

    # Deconv stage 4/stage 8
    x = deconv_block(x, 3, [128, 128, 64], stage=8, branch=branch, block='a')

    x = KL.Conv2D(512, (1, 1), name='output_layer_b', use_bias=True)(x)
    x = BatchNorm(axis=3, name='output_layer_b' + 'bn')(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv2D(2, (1, 1), name='output_layer_a', use_bias=True, activation='linear')(x)

    return x


############################################################
#  Data Generator
############################################################
def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,

    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]

            # Load image
            image = dataset.load_image(image_id)
            image = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                padding=config.IMAGE_PADDING)

            (height, width) = image.shape

            marginal = config.MARGINAL_PIXEL
            patch_size = config.PATCH_SIZE

            # create random point P within appropriate bounds
            y = random.randint(marginal, height - marginal - patch_size)
            x = random.randint(marginal, width - marginal - patch_size)
            # define corners of image patch
            top_left_point = (x, y)
            bottom_left_point = (x, patch_size + y)
            bottom_right_point = (patch_size + x, patch_size + y)
            top_right_point = (x + patch_size, y)

            four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
            perturbed_four_points = []
            for point in four_points:
                perturbed_four_points.append((point[0] + random.randint(-marginal, marginal),
                                              point[1] + random.randint(-marginal, marginal)))

            y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            # Two branches. The CNN try to learn the H and inv(H) at the same time. So in the first branch, we just compute the
            #  homography H from the original image to a perturbed image. In the second branch, we just compute the inv(H)
            H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
            H_inverse = np.linalg.inv(H)
            warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))

            img_patch_ori = image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
            img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1],
                                          top_left_point[0]:bottom_right_point[0]]

            point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
            diff_branch1 = point_transformed_branch1 - point
            diff_x_branch1 = diff_branch1[:, 0]
            diff_y_branch1 = diff_branch1[:, 1]

            diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
            diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

            pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                                                top_left_point[0]:bottom_right_point[0]]

            pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                                                top_left_point[0]:bottom_right_point[0]]

            pf_patch = np.zeros((config.PATCH_SIZE, config.PATCH_SIZE, 2))
            pf_patch[:, :, 0] = pf_patch_x_branch1
            pf_patch[:, :, 1] = pf_patch_y_branch1

            img_patch_ori = mold_image(img_patch_ori, config)
            img_patch_pert = mold_image(img_patch_pert, config)
            image_patch_pair = np.zeros((patch_size, patch_size, 2))
            image_patch_pair[:, :, 0] = img_patch_ori
            image_patch_pair[:, :, 1] = img_patch_pert

            # Init batch arrays
            if b == 0:
                batch_image_patch_pair = np.zeros((batch_size,) + (config.PATCH_SIZE, config.PATCH_SIZE, 2),
                                                 dtype=np.float32)
                batch_pf_patch = np.zeros((batch_size,) + (config.PATCH_SIZE, config.PATCH_SIZE, 2),
                                          dtype=np.float32)
            # Add to batch
            batch_image_patch_pair[b, :, :, :] = image_patch_pair
            batch_pf_patch[b, :, :, :] = pf_patch

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_image_patch_pair]
                outputs = [batch_pf_patch]

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

############################################################
#  Dense Perspective Class
############################################################

class DensePerspective():
    """Encapsulates the Perspective Network (PFNet) model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build PFNet architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image_patch_pair = KL.Input(shape=[config.PATCH_SIZE, config.PATCH_SIZE, 2], name="input_image_patch_pair")

        if mode == "training":
            x = pfnet_graph(input_image_patch_pair, "resnet50", branch=1, stage5=False)
            model =KM.Model(input_image_patch_pair, x, name='fcnresnet50')

        else:
            x = pfnet_graph(input_image_patch_pair, "resnet50", branch=1, stage5=False)
            model =KM.Model(input_image_patch_pair, x, name='fcnresnet50')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("pfnet"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum, config):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004, clipnorm=5.0)
		# Compile
        self.keras_model.compile(optimizer=optimizer, loss=smooth_l1_loss, metrics=[mace])


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/pfnet_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/pfnet\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "pfnet_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, augment=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE,
                                       augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
		    keras.callbacks.CSVLogger('training.csv', append=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        # self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM, self.config)

        if os.name is 'nt':
            workers = 0
        else:
            workers = min(self.config.BATCH_SIZE // 2, 0)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=200,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]   # (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return (images.astype(np.float32) - config.MEAN_PIXEL)


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
