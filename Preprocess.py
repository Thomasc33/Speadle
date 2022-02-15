"""
 This script is for preprocessing the raw dataset images into 
 usable training samples for our project
"""
import os
import cv2
import random
import Filters
import numpy as np
from scipy import ndimage


def loadDataset():
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
            shaped_image = cv2.resize(image, dsize=(128,128),interpolation = cv2.INTER_AREA)

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


dataset = loadDataset()

speed = '25'
rand_idx = random.randint(0, len(dataset[speed])-1)

image = dataset[speed][rand_idx]['image']
label = dataset[speed][rand_idx]['label']

gauss = ndimage.convolve(image, Filters.gaussian(3, 1.2))
edges = Filters.sobel_edges(gauss)

cv2.imshow("Edge-detected image", edges) 
cv2.waitKey(0)
cv2.destroyAllWindows()
        