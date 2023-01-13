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
#https://choosealicense.com/licenses/
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def image_read(path ):
    '''
    Funkcja zaczytująca plik graficzny
    wejscie:ściezka do pliku
    wyjscie: obraz RGB w rozmiarze 256x256x3 oraz wymiary oryginalnego obrazu
    '''
    img_ = cv2.imread(path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    original_shape = img_.shape
    return img_, original_shape   


def prediction_to_bbox(bboxes, image_grid):    
    '''
    Funkcja przetwarzająca wyjście modelu do zasięgów bounding_box.
    Copyright 2020 Matt Bast
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
    '''
    Funkcja wybiera podzbiór zasięgów ograniczających w malejącej kolejności według miary pokrycia Jaccarda poprzez
    usuwanie zasięgów, które w dużym stopniu pokrywają się z wcześniej wybranymi polami.
    '''
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
    '''
    Copyright 2020 Matt Bast
    '''
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
    bboxes = {}
    predictions = prediction_to_bbox(predictions, image_grid)
    bboxes = non_max_suppression(predictions, top_n=100)
    bboxes[:,2:4] = bboxes[:,2:4] - bboxes[:,0:2]
    return bboxes

def form_image_grid():    
    image_grid = np.zeros((32, 32, 4))
    # x, y, width, height
    rozmiar = 256
    
    cell = [0, 0, rozmiar / 32, rozmiar / 32] 
    for i in range(0, 32):
        for j in range(0, 32):
            image_grid[i,j] = cell

            cell[0] = cell[0] + cell[2]
        cell[0] = 0
        cell[1] = cell[1] + cell[3]
    return image_grid


def process_image(image_path, model, bbox_c):
    '''
    Funkcja wykorzystująca model sieci konwolucyjnej do predykcji
    wejscie:ściezka do pliku, model sieci konwolucyjnej
    wyjscie: obraz RGB w rozmiarze 256x256x3 oraz wymiary oryginalnego obrazu
    '''
    # read image and preprocess
    # Zmiana rozmiaru obrazu do 256x256 pikseli

    #Zaczytanie obrazu 
    img, orig_shape = image_read(image_path)
    img256 = cv2.resize(img, (256, 256))
    #h, w, _ = img.shape
    
    input_tensor = np.expand_dims(img256, 0)
    pred = model.predict(input_tensor / np.max( input_tensor))

    img_grid = form_image_grid()

    bbox = process_prediction(pred[0],  img_grid)
    
    # wyświetlenie detekcji w rozmiarze dostosowanym do rozmiarów wyswietlacza urządzenia
    img = cv2.resize(img, (orig_shape[1], orig_shape[0]))

    # Umieszczenie informacji o liczbie klosow na obrazie w stałym miejscu
    cv2.rectangle(img, (32, 32), (32 + 300, 32 - 100), (0, 0, 0), -1)
    
    img_ = cv2.putText(img, 
                       f"Liczba klosow: {bbox.shape[0]}",
                       (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (255, 255, 255), 1, cv2.LINE_AA)
    
    # Umieszczenie zasięgow na obrazie
    for b in range(bbox.shape[0]):
        
        x, y, width, height = bbox[b]
        
        x = round(x * orig_shape[0] / 256)
        y = round(y * orig_shape[1] / 256)

        width = round(width * orig_shape[0] / 256)
        height = round(height * orig_shape[1] / 256)
        
        ##umieszczenie zasięgow bounding-box na obrazie
        #bounding_box_color =  #czerwony kolor zasięgów
        bounding_box_color = [colors.to_rgb(bbox_c)[x] * 255 for x in range(3)]
        cv2.rectangle(img_,  (int(x), int(y)),
                              (int(x + width), 
                              int(y + height)), 
                              bounding_box_color, 2)
        
    # Przetworzenie finalne i opcjonalny zapis obrazu na lokalnym dysku
    img = cv2.resize(img, (int(orig_shape[1]*0.5), int(orig_shape[0]*0.5)))
    cv2.imwrite(image_path, cv2.cvtColor(img_, cv2.COLOR_RGB2BGR))
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return bbox.shape[0], image_path