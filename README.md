This repository contains all data and code related to term project on course "Introduction in Computer Vision"
at Skoltech. 

Goal of the project is to detect digits on water meter accurately.

Authors: Valerii Kornilov, Sergey Petrakov

Structure:

- Data: 
  - Train folder contains 20 pictures of water meters and y_true.txt is the file with correct readings 
  - Templates folder contains 5 templates of 5 different models of meters named model{i}.png and corresponding file boundaries{i}.txt with 2 coordinates (upper left and
    bottom right coordinates of bounding box for digits block: min_r, max_r ,min_c, max_c, where r means row, c - column)
  - DySyNet_state_dict.pt - weights of pretrained DiSyNet CNN model
  - mnist_net_99.pt - weights of pretrained MNISTNet CNN model
  - Digits recoginition folder contains digits pictures from which we create synthetical dataset

- Code:
  - Baseline.ipynb - notebook with an approach based on canny, hough circle transform, color matching, geometrical properties of image and CNN for classification
  - Keypoints.ipynb - approach based on BRISK key points detection and CNN for classification
  - Models.py - file with DiSyNet and MNISTNet CNNs
  - Digits_generation.ipynb - notebook with generation of synthetic dataset with digits (we'd like to have a CNN which could classify transition between digits on water meter correctly
    like 0/9, 1/2, 3/4 situations)
  - DiSyNet_train.ipynb - train DiSyNet model on synthetical dataset
  - MNISTNet_train.ipynb - train DiSyNet model on MNIST dataset
