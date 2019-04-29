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


import numpy as np
import utils
import random
import cv2
import model as modellib
import logging
import os
import time
import sys
import keras.backend as K
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO


from config import Config

import model as modellib
import tensorflow as tf


random.seed(a=1)
# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights files
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0300l1.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = '2014'

def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return (images.astype(np.float32) - config.MEAN_PIXEL)

class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 25

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 2


############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco



def data_generator_evaluation(dataset, config, shuffle=True, augment=True, random_rois=0,
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

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # Load image and mask
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
            warped_image = cv2.warpPerspective(image, np.linalg.inv(H), (image.shape[1], image.shape[0]))

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


            base_four_points = np.asarray([x, y,
                                           x, patch_size + y,
                                           patch_size + x, patch_size + y,
                                           x + patch_size, y])

            perturbed_four_points = np.asarray(perturbed_four_points)
            perturbed_base_four_points = np.asarray([perturbed_four_points[0, 0], perturbed_four_points[0, 1],
                                                     perturbed_four_points[1, 0], perturbed_four_points[1, 1],
                                                     perturbed_four_points[2, 0], perturbed_four_points[2, 1],
                                                     perturbed_four_points[3, 0], perturbed_four_points[3, 1]])
            # Init batch arrays
            if b == 0:
                batch_image_patch_pair = np.zeros((batch_size,) + (config.PATCH_SIZE, config.PATCH_SIZE, 2),
                                                 dtype=np.float32)
                batch_pf_patch = np.zeros((batch_size,) + (config.PATCH_SIZE, config.PATCH_SIZE, 2),
                                          dtype=np.float32)
                batch_base_four_points = np.zeros((batch_size, 8), dtype=np.float32)

                batch_perturbed_base_four_points = np.zeros((batch_size, 8), dtype=np.float32)

            # Add to batch
            batch_image_patch_pair[b, :, :, :] = image_patch_pair
            batch_pf_patch[b, :, :, :] = pf_patch
            batch_base_four_points[b, :] = base_four_points
            batch_perturbed_base_four_points[b, :] = perturbed_base_four_points
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_image_patch_pair]
                outputs = [batch_pf_patch]

                yield inputs, outputs, batch_base_four_points, batch_perturbed_base_four_points

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

def evaluate_PFNet(model, val_generator, limit=0, batch_size=0):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset

    # Limit to a subset
    assert batch_size > 0, "please make sure you the batchsize you entered larger than 0"

    steps = int(limit/batch_size)

    # def eucl_loss(x, y):
    #     l = K.sum(K.square(x - y)) / batch_size / 2
    #     return l
    # optimizer = keras.optimizers.Nadam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004,
    #                                    clipnorm=5.0)
    # # Compile
    # model.keras_model.compile(optimizer=optimizer, loss=eucl_loss, metrics=[mace])
    t_start = time.time()

    # Run detection
    n = -1
    total_mace = []
    print('Total steps', steps)
    while n < steps:
        print('Now step:', n + 1)
        n = n + 1
        X, Y_true, base_four_points, perturbed_base_four_points = next(val_generator)
        Y_true = np.asarray(Y_true).squeeze()
        Y_pred = model.keras_model.predict(X, batch_size=batch_size, verbose=1)
        mace_ = metric_paf(Y_true, Y_pred, config.PATCH_SIZE, base_four_points, perturbed_base_four_points)
        total_mace.append(mace_)

    final_mace = np.mean(total_mace)

    t_prediction = (time.time() - t_start)

    # Convert results to COCO format

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / limit))
    print("Total time: ", time.time() - t_start)
    print("MACE Metric: ", final_mace)

def metric_paf(Y_true, Y_pred, PATCH_SIZE, base_four_points, perturbed_base_four_points):
    # Compute the True H using Y_true
    assert (Y_true.shape == Y_pred.shape), "the shape of gt and pred should be the same"
    batch_size = Y_true.shape[0]


    mace_b = []
    
    for i in range(batch_size):
        Y_true_in_loop = Y_true[i, :, :, :]
        Y_pred_in_loop = Y_pred[i, :, :, :]
        base_four_points_in_loop = base_four_points[i, :]
        perturbed_base_four_points_in_loop = perturbed_base_four_points[i, :]

        delta_left_top_x = Y_true_in_loop[0, 0, 0]
        delta_left_bottom_x = Y_true_in_loop[127, 0, 0]
        delta_right_bottom_x = Y_true_in_loop[127, 127, 0]
        delta_right_top_x = Y_true_in_loop[0, 127, 0]

        delta_left_top_y = Y_true_in_loop[0, 0, 1]
        delta_left_bottom_y = Y_true_in_loop[127, 0, 1]
        delta_right_bottom_y = Y_true_in_loop[127, 127, 1]
        delta_right_top_y = Y_true_in_loop[0, 127, 1]

        gt_delta_four_point = np.asarray([(delta_left_top_x, delta_left_top_y),
                                   (delta_left_bottom_x, delta_left_bottom_y),
                                   (delta_right_bottom_x, delta_right_bottom_y),
                                   (delta_right_top_x, delta_right_top_y)])*256

        # define corners of image patch
        top_left_point = (base_four_points_in_loop[0], base_four_points_in_loop[1])
        bottom_left_point = (base_four_points_in_loop[2], base_four_points_in_loop[3])
        bottom_right_point = (base_four_points_in_loop[4], base_four_points_in_loop[5])
        top_right_point = (base_four_points_in_loop[6], base_four_points_in_loop[7])

        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        perturbed_top_left_point = (perturbed_base_four_points_in_loop[0], perturbed_base_four_points_in_loop[1])
        perturbed_bottom_left_point = (perturbed_base_four_points_in_loop[2], perturbed_base_four_points_in_loop[3])
        perturbed_bottom_right_point = (perturbed_base_four_points_in_loop[4], perturbed_base_four_points_in_loop[5])
        perturbed_top_right_point = (perturbed_base_four_points_in_loop[6], perturbed_base_four_points_in_loop[7])

        perturbed_four_points = [perturbed_top_left_point, perturbed_bottom_left_point, perturbed_bottom_right_point, perturbed_top_right_point]

        predicted_pf_x1 = Y_pred_in_loop[:, :, 0]
        predicted_pf_y1 = Y_pred_in_loop[:, :, 1]

        pf_x1_img_coord = predicted_pf_x1
        pf_y1_img_coord = predicted_pf_y1


        y_patch_grid, x_patch_grid = np.mgrid[0:config.PATCH_SIZE, 0:config.PATCH_SIZE]

        patch_coord_x = x_patch_grid + top_left_point[0]
        patch_coord_y = y_patch_grid + top_left_point[1]

        points_branch1 = np.vstack((patch_coord_x.flatten(), patch_coord_y.flatten())).transpose()
        mapped_points_branch1 = points_branch1 + np.vstack(
            (pf_x1_img_coord.flatten(), pf_y1_img_coord.flatten())).transpose()


        original_points = np.vstack((points_branch1))
        mapped_points = np.vstack((mapped_points_branch1))

        H_predicted = cv2.findHomography(np.float32(original_points), np.float32(mapped_points), cv2.RANSAC, 10)[0]

        predicted_delta_four_point = cv2.perspectiveTransform(np.asarray([four_points], dtype=np.float32),
                                                                  H_predicted).squeeze() - np.asarray(perturbed_four_points)

        result = np.mean(np.linalg.norm(predicted_delta_four_point, axis=1))
        mace_b.append(result)

    __mace = np.mean(mace_b)
    return __mace

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evalute PFNet on COCO dataset.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=5000,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=5000), as mentioned in the paper')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("-------------------------PFNet evaluation on the COCO dataset-------------")
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations

    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 50

    config = InferenceConfig()
    config.display()

    # Create model
    model = modellib.DensePerspective(mode="inference", config=config, model_dir=args.logs)

    # Load weights
    model_path = args.model
    print("Loading weights ", model_path)
    # model.load_weights(model_path, by_name=True)
    model.load_weights(model_path, by_name=True)

    # Validation dataset
    dataset_val = CocoDataset()
    coco = dataset_val.load_coco(args.dataset, "val", year=args.year, return_coco=True,
                                 auto_download=args.download)
    dataset_val.prepare()
    val_generator = data_generator_evaluation(dataset_val, config, shuffle=True,
                                              batch_size=config.BATCH_SIZE,
                                              augment=False)
    print("Running COCO evaluation on {} images regarding PFNet performance.".format(args.limit))
    evaluate_PFNet(model, val_generator, limit=int(args.limit), batch_size=config.BATCH_SIZE)

