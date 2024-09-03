"""
Copyright 2020 Matt Bast

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import sys, os
import cv2
from sklearn import preprocessing
import os
import tensorflow as tf
from matplotlib import colors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def image_read(path):
    '''
    Reads an image file and converts it to RGB format.

    This function reads an image from the specified file path, converts the image 
    from BGR (default in OpenCV) to RGB format, and returns the image along with its 
    original dimensions.

    Parameters:
    - path: str
        The path to the input image file.

    Returns:
    - img_: numpy.ndarray
        The image read from the file, converted to RGB format.
    - original_shape: tuple
        The dimensions of the original image (height, width, channels).
    '''
    img_ = cv2.imread(path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    original_shape = img_.shape
    return img_, original_shape   

def prediction_to_bbox(bboxes, image_grid):    
    '''
    Copyright 2020 Matt Bast

    Converts model predictions to bounding box coordinates.

    This function processes the output from a neural network model to convert it into 
    bounding box coordinates for object detection. The function takes the raw output 
    predictions (in the form of bounding box center coordinates and dimensions) and 
    deserializes them to fit the image grid. The output bounding boxes are represented 
    by their top-left and bottom-right corner coordinates.

    Parameters:
    - bboxes: numpy.ndarray
        A 3D numpy array of shape (32, 32, 10) representing the model's raw predictions.
        The array contains:
        - Two sets of bounding boxes for each cell in a 32x32 grid (each set with 4 values: 
          [x_center, y_center, width, height]).
    - image_grid: numpy.ndarray
        A 3D numpy array representing the image grid, containing the scale and offset for 
        each cell in the grid. Used to transform bounding box coordinates to the original 
        image scale.

    Returns:
    - numpy.ndarray
        A 3D numpy array containing the descaled bounding boxes in the format 
        [x_min, y_min, x_max, y_max].
    '''  

    bboxes = bboxes.copy()
    
    im_width = (image_grid[:,:,2] * 32)
    im_height = (image_grid[:,:,3] * 32)
    
    # descale x,y
    bboxes[:,:,1] = (bboxes[:,:,1] * image_grid[:,:,2]) + image_grid[:,:,0]
    bboxes[:,:,2] = (bboxes[:,:,2] * image_grid[:,:,3]) + image_grid[:,:,1]
    bboxes[:,:,6] = (bboxes[:,:,6] * image_grid[:,:,2]) + image_grid[:,:,0]
    bboxes[:,:,7] = (bboxes[:,:,7] * image_grid[:,:,3]) + image_grid[:,:,1]
    
    # descale width,height
    bboxes[:,:,3] = bboxes[:,:,3] * im_width 
    bboxes[:,:,4] = bboxes[:,:,4] * im_height
    bboxes[:,:,8] = bboxes[:,:,8] * im_width 
    bboxes[:,:,9] = bboxes[:,:,9] * im_height
    
    # centre x,y to top left x,y
    bboxes[:,:,1] = bboxes[:,:,1] - (bboxes[:,:,3] / 2)
    bboxes[:,:,2] = bboxes[:,:,2] - (bboxes[:,:,4] / 2)
    bboxes[:,:,6] = bboxes[:,:,6] - (bboxes[:,:,8] / 2)
    bboxes[:,:,7] = bboxes[:,:,7] - (bboxes[:,:,9] / 2)
    
    # width,heigth to x_max,y_max
    bboxes[:,:,3] = bboxes[:,:,1] + bboxes[:,:,3]
    bboxes[:,:,4] = bboxes[:,:,2] + bboxes[:,:,4]
    bboxes[:,:,8] = bboxes[:,:,6] + bboxes[:,:,8]
    bboxes[:,:,9] = bboxes[:,:,7] + bboxes[:,:,9]
    return bboxes


def non_max_suppression(predictions, top_n):
    """
    Copyright 2020 Matt Bast
    
    Selects a subset of bounding boxes based on non-maximum suppression.

    This function performs non-maximum suppression (NMS) to select the top bounding boxes 
    from a set of predicted bounding boxes. The bounding boxes are selected in descending 
    order of their scores (probabilities) while removing boxes that have a high overlap 
    (measured by the Jaccard index or Intersection over Union - IoU) with already selected boxes.

    Parameters:
    - predictions: numpy.ndarray
        A 3D numpy array of shape (32, 32, 10) representing model predictions. The array contains:
        - Two sets of bounding box coordinates (each with 4 values) for each cell in a 32x32 grid.
        - Corresponding confidence scores for each bounding box.
    - top_n: int
        The maximum number of bounding boxes to select after applying non-maximum suppression.

    Returns:
    - bboxes: numpy.ndarray
        A 2D numpy array containing the selected bounding boxes after applying non-maximum suppression.
        Each bounding box is represented in the format [x1, y1, x2, y2].
    """
    
    probabilities = np.concatenate((predictions[:,:,0].flatten(), 
                                    predictions[:,:,5].flatten()), axis=None)
    
    first_anchors = predictions[:,:,1:5].reshape((32*32, 4))
    second_anchors = predictions[:,:,6:10].reshape((32*32, 4))
    
    bboxes = np.concatenate( (first_anchors, second_anchors), axis = 0)
    
    bboxes = switch_x_y(bboxes)
    #bboxes, probabilities = select_top(probabilities, bboxes, top_n=top_n)
    
    top_indices = tf.image.non_max_suppression(
        boxes = bboxes, 
        scores = probabilities, 
        max_output_size = top_n, 
        iou_threshold = 0.3,
        score_threshold = 0.3).numpy()
    
    bboxes, probabilities = bboxes[top_indices], probabilities[top_indices]

    bboxes = switch_x_y(bboxes)
    
    return bboxes


def switch_x_y(bboxes):
    """
    Switches the x and y coordinates of bounding boxes.

    This function swaps the x and y coordinates of bounding boxes in a given array.
    The input array `bboxes` is expected to have bounding boxes in the format 
    [x1, y1, x2, y2], where (x1, y1) represents the top-left corner and (x2, y2)
    represents the bottom-right corner. After switching, the bounding boxes will be 
    in the format [y1, x1, y2, x2].

    Parameters:
    - bboxes: numpy.ndarray
        A 2D numpy array where each row represents a bounding box in the format 
        [x1, y1, x2, y2].

    Returns:
    - numpy.ndarray
        The modified array with x and y coordinates swapped for each bounding box.
    """
    x1 = bboxes[:,0].copy()
    y1 = bboxes[:,1].copy()
    x2 = bboxes[:,2].copy()
    y2 = bboxes[:,3].copy()
    bboxes[:,0] = y1
    bboxes[:,1] = x1
    bboxes[:,2] = y2
    bboxes[:,3] = x2
    return bboxes


def process_prediction(predictions, image_grid):
    """
    Processes model predictions to generate bounding boxes.

    This function takes raw predictions from a model and processes them to 
    produce bounding boxes using a predefined image grid. It first converts 
    the raw predictions into bounding box format using `prediction_to_bbox()`, 
    then applies non-maximum suppression to filter out overlapping bounding 
    boxes and retain only the top bounding boxes based on confidence scores. 
    Finally, it adjusts the bounding box coordinates to represent the width 
    and height.

    Parameters:
    - predictions: array-like
        Raw predictions from a machine learning model. These predictions 
        typically include bounding box coordinates and confidence scores.
    - image_grid: numpy.ndarray
        A grid representing the spatial layout of the image, used to map 
        predictions to their corresponding locations in the image.

    Returns:
    - bboxes: numpy.ndarray
        An array of processed bounding boxes after non-maximum suppression. 
        Each bounding box is represented in the format [x, y, width, height].
    """
    bboxes = {}
    predictions = prediction_to_bbox(predictions, image_grid)
    bboxes = non_max_suppression(predictions, top_n = 100)
    bboxes[:,2:4] = bboxes[:,2:4] - bboxes[:,0:2]
    return bboxes

def form_image_grid(im_size = 256):   
    """
    Creates a grid of image cells for a specified image size.

    This function generates a 32x32 grid, where each cell contains information 
    about its position and size in the format [x, y, width, height]. The image 
    grid is designed to work with a given image size and divides the image 
    into equal cells.

    Parameters:
    - im_size: int, optional
        The size of the image (in pixels) that will be divided into the grid.
        Default is 256 pixels.

    Returns:
    - image_grid: numpy.ndarray
        A 32x32x4 NumPy array where each entry represents a cell in the grid. 
        Each cell contains four values [x, y, width, height], indicating the 
        position and size of the cell within the image grid.
    """
    image_grid = np.zeros((32, 32, 4))
    # x, y, width, height
    cell = [0, 0, im_size / 32, im_size / 32] 
    for i in range(0, 32):
        for j in range(0, 32):
            image_grid[i,j] = cell

            cell[0] = cell[0] + cell[2]
        cell[0] = 0
        cell[1] = cell[1] + cell[3]
    return image_grid


def process_image(image_path, model, bbox_c):
    '''
    Uses a convolutional neural network model to make predictions on an image.

    This function reads an image from the specified file path, resizes it to 256x256 pixels,
    and uses a convolutional neural network (CNN) model to predict bounding boxes of detected objects.
    The predictions are processed to display bounding boxes and object count on the original image size.
    The final processed image is saved back to the specified file path.

    Parameters:
    - image_path: str
        The path to the input image file.
    - model: keras.Model
        The convolutional neural network model used for making predictions.
    - bbox_c: str
        The color code for bounding boxes, specified in a format recognized by matplotlib.colors.

    Returns:
    - int
        The number of detected objects (bounding boxes).
    - str
        The path to the processed image file saved on disk.
    '''
    # read image and preprocess
    img, orig_shape = image_read(image_path)
    img256 = cv2.resize(img, (256, 256))
    #h, w, _ = img.shape
    
    input_tensor = np.expand_dims(img256, 0)
    pred = model.predict(input_tensor / np.max( input_tensor ))

    img_grid = form_image_grid()

    bbox = process_prediction(pred[0],  img_grid)
    
    # Display of detection in a size adapted to the size of the device display
    img = cv2.resize(img, (orig_shape[1], orig_shape[0]))

    # Putting information about the number of closhes in the image in a fixed place
    cv2.rectangle(img, (32, 32), (32 + 300, 32 - 100), (0, 0, 0), -1)
    
    img_ = cv2.putText(img, 
                       f"Liczba klosow: {bbox.shape[0]}",
                       (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (255, 255, 255), 1, cv2.LINE_AA)
    
    # Placing bboxes in the image
    for b in range(bbox.shape[0]):
        
        x, y, width, height = bbox[b]
        
        x = round(x * orig_shape[0] / 256)
        y = round(y * orig_shape[1] / 256)

        width = round(width * orig_shape[0] / 256)
        height = round(height * orig_shape[1] / 256)
        
        ## Placement of bounding-box ranges on the image

        bounding_box_color = [colors.to_rgb(bbox_c)[x] * 255 for x in range(3)]
        cv2.rectangle(img_,  (int(x), int(y)),
                              (int(x + width), 
                              int(y + height)), 
                              bounding_box_color, 2)
        
    # Final processing and optional image storage on local disk
    img = cv2.resize(img, (int(orig_shape[1]*0.5), int(orig_shape[0]*0.5)))
    cv2.imwrite(image_path, cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return bbox.shape[0], image_path
