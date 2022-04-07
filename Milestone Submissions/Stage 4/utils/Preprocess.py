"""
 This script is for preprocessing the raw dataset images into 
 usable training samples for our project
"""
import os
import cv2
import random
from utils import Filters
import numpy as np
from scipy import ndimage


def loadDataset(image_dims=[256,256]):
    """
    Load raw speed limit dataset into a dictionary (object)

    The resulting dataset will have 7 keys, one for each speed limit:
    - '25'
    - '30'
    - '35'
    - '40'
    - '45'
    - '50'
    - '60'
    
    Each key pairs to an array of image objects with two keys:
    - 'image' : RGB image matrix
    - 'label' : The label for this image (speed limit)

    Returns:
        dict: speed limit sign dataset
    """
    data_dir = os.listdir('./data')
    dataset = {}
        
    # Loop through each speed folder in the raw dataset
    for speed_limit in data_dir:
        # Get speed limit directory
        speed_dir = os.listdir('./data/' + str(speed_limit))
        
        # Object for storing all the data samples for this speed limit
        speed_images = []
        
        # Loop through and read each image in this speed's dataset
        for image_path in speed_dir:
            image = np.array(cv2.imread("data/" + str(speed_limit) + "/" + str(image_path))).astype(np.float64)
            
            # Reshape image
            shaped_image = cv2.resize(image, dsize=(image_dims[0],image_dims[1]),interpolation = cv2.INTER_AREA)

            # Convert image from RGB to grayscale  
            shaped_image /= 255.0 # Normalize image      
            shaped_image = Filters.grayscale(shaped_image)

            # Create object to store image and its label
            image_data = {
                "image": shaped_image,
                "label": speed_limit
            }
            
            # Add image data to speed images
            speed_images.append(image_data)
            
            
        # Add speed images to the main dataset
        dataset[speed_limit] = speed_images
        
    return dataset
    

# Split data into training, validation, and testing sets
def split_dataset(dataset, train_pct, test_pct, valid_pct):
  train_data = {}
  test_data = {}
  valid_data = {}

  # Loop through keys (speed limits) in dataset
  for i,key in enumerate(dataset):
    # Compute number of samples for each set
    n_train_samples = int(len(dataset[key])*train_pct)
    n_test_samples = int(len(dataset[key])*test_pct)
    n_valid_samples = int(len(dataset[key])*valid_pct)

    # Populate datasets
    train_data[key] = dataset[key][:n_train_samples]
    test_data[key] = dataset[key][n_train_samples+1:n_train_samples+1+n_test_samples]
    valid_data[key] = dataset[key][n_train_samples+n_test_samples+2:]


  # I'm sure there's a better way to do this but it works
  _train = []
  _test = []
  _valid = []

  # Reformat data into a single array
  for key in train_data:
    for sample in train_data[key]:
      _train.append(sample)
    for sample in test_data[key]:
      _test.append(sample)
    for sample in valid_data[key]:
      _valid.append(sample)


  return _train, _test, _valid

# Reshape data for use as inputs into the model
def reshape_data(data, image_shape=64):
  return np.reshape(data, [-1, image_shape, image_shape])

# This allows us to simply import dataset from Preprocess in any file
#dataset = loadDataset()