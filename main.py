'''
MIT License
Copyright 2023 Dominika Nowa; Katatrzyna Krupa;Dominik Mielczarek
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
parser.add_argument('--weights_dir', dest='weights_dir', type=str, default="/media/H/tree_quant/Dominik/wheat/weights/yolov5.hdf5", help='Lokalizacja wag modelu w formacie .hdf5')
parser.add_argument('--temp_dir', dest='temp_dir', type=str, default="static/temp/", help='Lokalizacja ścieżki tymczasowej - miejsce zapisu detekcji')
parser.add_argument('--gpu', dest='gpu', type=str, default="", help='Użycie procesora graficznego GPU brak-"", jedno urządzenie - "0", dwa urządzenia -"0,1" itd')
parser.add_argument('--bbox_color', dest='bbox_colour', type=str, default="red", help='Kolor zasięgów - r-czerwony, g-zielony, b-niebieski')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Inicjalizacja aplikacji sieciowej
app = Flask(__name__) #, static_folder='static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
 
cache = Cache()
cache.init_app(app=app, 
               config={"CACHE_TYPE": "filesystem",
                       'CACHE_DIR': '/temp'})

# Inicjalizacja i skompilowanie grafu modelu YOLO w wersji 3
model = YOLO_v3()
model.compile(loss = None,
              optimizer = 'adam')

# wczytanie wag modelu
model.load_weights(args.weights_dir)

@app.route('/')
def index():
    '''
    render schematu html do aplikacji
    '''
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/upload', methods=['POST'])
def upload():
    '''
    Funkcja zaczytuje plik wskazany przez użytkownika 
    '''
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
    '''
    Obliczenie plonowania
    '''
    #render_template('index.html')
    mtn = float(request.form['mtn'])
    slzk = float(request.form['slzk'])
    strata = float(request.form['strata'])
    
    ok = cache.get("ok")
    im_path = cache.get("image_path")

     # wspolczynnik strat - stała
    plon = ( (4*ok * slzk * mtn) / 1e+4 )*(1 - (strata/100))

    print(cache.get("img_width"), cache.get("img_height"))
    return render_template("index.html", mtn = mtn,
                                         slzk = slzk, 
                                         strata = strata, 
                                         plon_w = round( float(plon), 3),  
                                         image_path = im_path )

# Uruchomienie aplikacji
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True) # set debug true to load reload server auto on changes
    
