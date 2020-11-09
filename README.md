# DeepDistance: A Multi-task Deep Regression Model for Cell Detection in Inverted Microscopy Images

This code is the implementation of a multi-task learning framework called DeepDistance that we proposed for the detection of live cells in inverted microscopy images. This DeepDistance framework proposes to concurrently learn two distance metrics, where the primary one is learned in regard to the main cell detection task and the secondary distance is learned for the purpose of increasing the generalization ability of the main task. To this end, it constructs a fully convolutional network and end-to-end learns two distance maps at the same time, sharing high-level feature representations at the various layers of this network, in the context of multi-task learning.

NOTE: The following source codes and executable are provided for research purposes only. The authors have no responsibility for any consequences of use of these source codes and the executable. If you use any part of these codes, please cite the following paper.
>C. F. Koyuncu, G. N. Gunesli, R. Cetin-Atalay, and C. Gunduz-Demir, "DeepDistance: A multi-task deep regression model for cell detection in inverted microscopy images," Medical Image Analysis, 63 101720 (2020).

Please contant Can Fahrettin Koyuncu at canfkoyuncu@gmail.com for further questions.

This repository contains five scripts:

* ***cellDetection.m***:  This Matlab code is to find the cell locations when the inner distance map of an image estimated by the DeepDistance model (or its extended version) is given.
* ***calculateDistances.m***:  This Matlab code is to calculate the inner and normalized outer distances for an image when its cell annotations are given. This code is to prepare the regression outputs for a training image.
* ***deepDistanceModels.py***:  This Python code includes the network architectures for the DeepDistance model as well as its extended version. It also includes the parameter settings used in these architectures.
* ***train.py***:  This Python code includes the function calls to train the DeepDistance model and its extended version. It also includes the code to prepare training/validation patches from a set of training images. 
* ***estimateDistances.py***: This Python code includes the function call to estimate the distance maps for a given image with a trained DeepDistance model or its extended version. 
