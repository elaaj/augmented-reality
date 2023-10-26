# augmented-reality
Computer vision project focused on augmented reality using marker detector and pose estimation.

## Table of Contents

- [Requirements](#Requirements)
- [Description](#Description)
- [Installation](#Installation)
- [Usage](#usage)


## Requirements

This repository contains the source code for Assignment 3 of Geometric and 3D Computer Vision course. It aims at performing an augmented reality by 
using marker point coordinates detected in the previous assignment, repository [polygonal-marker-detector](https://github.com/elaaj/polygonal-markers-detector/blob/main/README.md#polygonal-markers-detector), to compute marker pose and project 
a virtual cube on top of a rotating object using augmented reality techniques. 

## Description:

The **main.ipynb** jupyter notebook contains the code to track objects in a video, extract camera pose, and overlay a 3D cube on the object in each frame of the video. The *intrinsic parameters K* and the *distortion coefficients* are already provided: 
- *K*:
  [[1.66750771e+03 0.00000000e+00 9.54599045e+02]
  [0.00000000e+00 1.66972683e+03 5.27926123e+02]
  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
- *distortion*:
  [ 1.16577217e-01 -9.28944623e-02
  7.15149511e-05 -1.80025974e-03
  -1.24761932e-01]


## Installation

Before of running the main notebook, there are some requirements:
- Python 3.
- The following python modules:
```bash
pip install opencv-python
pip install numpy
```
- The dataset folder, which can be downloaded [here](https://github.com/elaaj/binary-video-segmentation/tree/main/data).

## Usage

The main notebook can be run from any IDE which supports Jupyter Notebooks.

```bash
git clone https://github.com/elaaj/augmented-reality
```
