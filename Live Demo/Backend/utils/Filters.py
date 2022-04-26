"""
    Various image filters
"""
import cv2, math
import numpy as np
from scipy import ndimage

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale

    Args:
        image (np.ndarray): RGB image

    Returns:
        np.ndarray: Grayscale image
    """
    
    return np.mean(image, -1)


def gaussian(kernel_size: int, sigma: int):
    """
    Generate a Gaussian filter kernel

    Args:
        kernel_size (int): Kernel size (i.e. Size-x-Size)
        sigma (int): filter strength
    """
    
    # Create a mesh grid that is 2x+1 the size
    x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
    
    # Compute the first part of the filter
    a = 1 / (2.0 * np.pi * sigma**2)
    
    # Compute exponential part of the filter
    b = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    
    # Compute and return kernel
    return a*b
    

def sobel_edges(image: np.ndarray):
    """
    Apply Sobel edge detection to an image

    Args:
        image (np.ndarray): 
    """
    # Create horizontal and vertical sobel kernels
    h_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float64)
    
    v_kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).astype(np.float64)


    # Find egdes
    h_edges = ndimage.convolve(image, h_kernel)
    v_edges = ndimage.convolve(image, v_kernel)
    
    # Combine horizontal and vertical to make entire sobel image
    return np.sqrt(np.square(h_edges) + np.square(v_edges))


def hardline_filter(edges, cutoff):
    m = -1

    # Loop through image 
    for row in edges:
      # Increment row index
      m += 1
      n = -1
      for col in row:
        # Increment column index
        n += 1
        
        if col < cutoff:
          edges[m][n] = 0
        else:
          edges[m][n] = 1

    return edges

def inverse_hardline_filter(edges, cutoff):
    m = -1

    # Loop through image 
    for row in edges:
      # Increment row index
      m += 1
      n = -1
      for col in row:
        # Increment column index
        n += 1
        
        if col > cutoff:
          edges[m][n] = 0
        else:
          edges[m][n] = 1

    return edges
