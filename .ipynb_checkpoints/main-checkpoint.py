'''
MIT License
Copyright 2023 Dominik Mielczarek
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
'''
from flask import Flask, render_template, request, redirect,url_for,send_from_directory
from src import detect
import tensorflow as tf
import sys, os
sys.path.append('./src')
import matplotlib.pyplot as plt
from model_f import *
import argparse
from flask_caching import Cache

# Instantiate the cache
parser = argparse.ArgumentParser(add_help = True)
parser.add_argument('--weights_dir', dest='weights_dir', type=str, default="./yolov5.hdf5", help='Location of model weights in format .hdf5')
parser.add_argument('--temp_dir', dest='temp_dir', type=str, default="static/temp/", help='Location of the temporary path - the diretory, where detection is recorded')
parser.add_argument('--gpu', dest='gpu', type=str, default="", help='GPU usage none-"", one gpu - "0", two gpus -"0,1" itd')
parser.add_argument('--bbox_color', dest='bbox_colour', type=str, default="red", help='b box color - r-red, g-green, b-blue')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Network application initialization
app = Flask(__name__) #, static_folder='static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
 
cache = Cache()
cache.init_app(app=app, 
               config={"CACHE_TYPE": "filesystem",
                       'CACHE_DIR': '/temp'})

# Initialize and compile the graph of the YOLO model version 3
model = YOLO_v3()

# loading model weights
model.load_weights(args.weights_dir)

@app.route('/')
def index():
    """
    Renders the main HTML template for the web application.

    This function returns the rendered HTML content for the main page of the web application.
    It uses Flask's `render_template` function to load and render the 'index.html' file.

    Parameters:
    - None

    Returns:
    - str: Rendered HTML content of 'index.html'.
    """
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """
    Serves the favicon for the web application.

    This function returns the favicon.ico file located in the 'static' directory of the 
    Flask application's root path. It is used to provide the favicon to the browser 
    when the application is accessed.

    Parameters:
    - None

    Returns:
    - Response: 
        A Flask response object that sends the favicon.ico file from the 'static' 
        directory with the appropriate MIME type for an icon.
    """
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/upload', methods=['POST'])
def upload():
    """
    Handles file upload from the user and processes the image for object detection.

    This function is responsible for receiving a file uploaded by the user via an HTML form.
    The uploaded file is saved to a temporary directory, and an object detection model is
    applied to the image. The function then stores the results of the detection in a cache
    and returns the URL of the processed image.

    Parameters:
    - None (input is retrieved directly from the form via the request object).

    Returns:
    - str: URL of the processed image file on the server.
    """
    render_template("index.html", visibility = "hidden")
    import glob, os, os.path
    with app.app_context():
        cache.clear()

    image_path = None
    file = request.files['file']
    filepath = f'{args.temp_dir}/{file.filename}'
    file.save(filepath) # save to directory

    # przeprowadzenie predykcji
    ok, image_path = detect.process_image(filepath, model, args.bbox_colour)

    cache.set("ok", ok)
    cache.set("image_path", image_path)

    return f"{request.url_root}{filepath}"


@app.route('/', methods=['POST'])
def my_form_post():
    """
    Calculates crop yield based on input parameters from a web form.

    This function retrieves user inputs from an HTML form using the POST method.
    It then calculates the crop yield using these inputs and a constant loss coefficient.
    The calculated yield, along with other input values, is rendered back to the user
    through an HTML template.

    Parameters:
    - None (inputs are retrieved directly from the form via the request object).

    Returns:
    - str: Rendered HTML page with the calculated crop yield and input values.
    """
    #render_template('index.html')
    mtn = float(request.form['mtn'])
    slzk = float(request.form['slzk'])
    loss = float(request.form['loss'])
    
    ok = cache.get("ok")
    im_path = cache.get("image_path")

     # wspolczynnik strat - stała
    plon = ( (4*ok * slzk * mtn) / 1e+4 )*(1 - (loss/100))

    print(cache.get("img_width"), cache.get("img_height"))
    return render_template("index.html", mtn = mtn,
                                         slzk = slzk, 
                                         strata = loss, 
                                         plon_w = round( float(plon), 3),  
                                         image_path = im_path )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
    
