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
import time
import numpy as np
import sys
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO


from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 32

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


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Perspective Network on COCO.')

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
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)

    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)

    # Configurations

    config = CocoConfig()

    config.display()

    # Create model
    model = modellib.DensePerspective(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load

    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    if args.model.lower() == 'last':
        model.load_weights(model_path, by_name=True)

    # Train or evaluate

    dataset_train = CocoDataset()
    dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CocoDataset()

    dataset_val.load_coco(args.dataset, "val", year=args.year, auto_download=args.download)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***

    # Training - Stage 1
    print("Training network using normal LR")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='all')
    print("Training network using LR/10")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=80,
                layers='all')
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers using LR/100")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=120,
                layers='all')

    print("Fine tune all layers using LR/100")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 1000,
                epochs=160,
                layers='all')


