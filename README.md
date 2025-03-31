# Team Challenge group 3 - Automatic C-arm Angle Algorithm

## Overview
This repository contains code and resources our automatic C-arm angle algorithm. The goal of this project is to develop a method for automatically determining the rotation angles of a C-arm to obtain a true AP fluoroscopy image of a target vertebra. This README will guide you through the setup, usage, and structure of the repository.

## Table of Contents
- Overview
- Table of Contents
- Dataset
- Requirements
- Usage

## Dataset
The dataset used for this project consisted of five 3D abdominal CT volumes (DICOM format) of healthy children whose parents have scoliosis. The data is not publicly available on this GitHub repository. Manual segmentations of each vertebra were made and are available in the boneMRI folder. The DICOM image folders can be placed inside the boneMRI folder for the code to function as intended.

## Requirements
- CustomTkinter
- Matplotlib
- Napari (optional as 3D viewer)
- Nibabel
- Numpy
- OpenCV
- PIL
- Scikit-learn
- SciPy
- SimpleITK
- Tkinter

## Usage
The code is divided into Python files and Jupyter Notebook files. The Python files contain the main functions that are used to implement our algorithm and the Jupyter Notebook files contain the implementation of this code along with code for visualization and testing purposes. Below, we quickly explain the code that can be found in all the files:

- `GUI.py` contains the customtkinter class for the graphical user interface of our algorithm. 
- `template_matching.py` contains functions for the template matching, which optimizes the rotation angles needed to go from an arbitrarily rotated radiograph to the true AP radiograph.
- `image_utils.py` contains common functionalities needed to transform the image data or to compute bounding boxes / PCA on the vertebra segmentations.

- `GUI.ipynb` can be used to start the graphical user interface. It should be relatively straight-forward to use and is the easiest way to experiment with the algorithm.
- `Template_matching.ipynb` is a notebook that was primarily used to test the code of the template matching algorithm.
- `load_data.ipynb` is a notebook where we experimented with the data and performed the first calculations of the boundings boxes / PCA.