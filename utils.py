"""

    Utility functions

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from GoogleImageScrapper import GoogleImageScrapper

def loadMNIST() -> dict:
    """
    Load the MNIST dataset. This dataset consists of 60,000 training images
    of hand-written digits between 0 and 9. As well as 10,000 testing images
    which will be further split into a testing and validation set.
    
    Currently, this is a 70/30 split. Meaning 7,000 testing images and 3,000
    validation images.

    #### Returns:
        dict: dictionary object containing all of the training,
                testing, and validation images in the following format:
                
                dict = {
                    "train_images": tuple,
                    "train_labels": tuple,
                    "test_images": tuple,
                    "test_labels": tuple,
                    "validation_images": tuple,
                    "validation_labels": tuple
                }
    """
    
    print("Loading MNIST dataset...\n")
    
    # Load training and testing images from tensorflow dataset
    [train_images, train_labels], [test_images, test_labels] = tf.keras.datasets.mnist.load_data()
    
    # Convert to lists
    train_images = list(train_images)
    test_images = list(test_images)
    train_labels = list(train_labels)
    test_labels = list(test_labels)
    
    
    # Split test_images into validation/test datasets
    validation_images = [test_images.pop(i) for i in range(3000)]   # Extract first 30% of testing images to be the validation data
    validation_labels = [test_labels.pop(i) for i in range(3000)]   # Extract corresponding labels
    
    print(f"MNIST DATASET LOADED:\n\tTraining samples: {len(train_images)}\n\tTesting samples: {len(test_images)}\n\tValidation samples: {len(validation_images)}")
    
    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "validation_images": validation_images,
        "validation_labels": validation_labels
    }

def scrapeImages(nImages: int):
    """
    Scrape and save training/testing images from google images

    Args:
        nImages (int): Number of images of each mph value to scrape 
        
    NOTE: mph values are from 5-85
    """
    mph_range = np.arange(start=5, stop=90, step=5)

    searches = [str(mph_range[i]) + "mph" for i in range(mph_range.shape[0])]
    
    for search in searches:
        gis = GoogleImageScrapper.GoogleImageScraper("GoogleImageScrapper/webdriver/chromedriver", "data\\", 
                                             search_key=search, number_of_images=nImages, headless=True)

        urls = gis.find_image_urls()
        gis.save_images(urls)


