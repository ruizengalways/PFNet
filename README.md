***
# Perspective Field Network
Code Repo of 'Rethinking Planar Homography Estimation Using Perspective Fields'

Tensorflow/Keras implementation for reproducing Perspective Network (PFNet) results in the paper [Rethinking Planar Homography Estimation Using Perspective Field](https://eprints.qut.edu.au/126933/) by Rui Zeng, Simon Denman, Sridha Sridharan, Clinton Fookes.
***
### COCO Dataset
- Please refer to [Common Objects in Context](http://cocodataset.org/#home) to download the dataset used in the paper.
***
### Trained Weights.
- Download our trained weights from [provisionally trained weights](https://www.dropbox.com/s/dk29bo0ml6ao7gc/pfnet_0200.h5?dl=0) and put it in the root directory

***
### Dependencies
python 3.6

- Tensorflow >= 1.5.0
- Keras >= 2.2.0
- Opencv >= 3.0.0

In addition, please add the project folder to PYTHONPATH and `pip install` the packages if `ImportError: No module named xxx` error message occur.
***
### Training
- Train a PFNet model on the COCO dataset from scratch:
  -  `python train.py --dataset=/home/COCO`

***
### Evaluation
- Evaluate the `*.h5` model checkpoint
  - `python evaluate.py --dataset=/home/COCO --model=./pfnet.h5`

***
### Citing PFNet
If you find PFNet useful in your research, please consider citing:
***

```
@inproceedings{zeng18rethinking,
  author    = {Rui Zeng and Simon Denman and Sridha Sridharan and Clinton Fookes},
  title     = {Rethinking Planar Homography Estimation Using Perspective Fields},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2018},
}
```
