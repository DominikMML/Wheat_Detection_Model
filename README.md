# **Wheat Detection Model - WDM**
![Logo](https://drive.google.com/uc?export=view&id=1xwnJd3MACXy-L-Rs1TBoWTd3J1RJ7-Se)

## Characteristics
Yield estimation based on in-situ photographic documentation.
The software makes it easy to determine the number of ears in a photo taken in the field.
The program is designed to streamline and speed up field work during in-situ inspections.
The program estimates yields based on plant density(number of plants in a 25x25cm measuring frame).
When the plant density is too high, the plants develop smaller (shorter) ears, which contain finer grain, with lower 1,000 grain weight and density.
This means that in such a case the yield will be lower and the quality will not be the best. The susceptibility of plants to lodging is also increased.
If, on the other hand, the ear density is too low, longer ears containing larger grains will not allow a satisfactory yield.
It is also important that the ear density be uniform for the entire plantation. 

![Logo](https://drive.google.com/uc?export=view&id=16BPGdp1Nu4a_PHs6a8kYxbyVXT6bSwpE)

### License terms and conditions
Implementation of the model (YOLO_v3) used under the license: 
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)\
No changes have been made to the scripts released under the Apache 2.0 license. 

Architecture of the detection model according to the article by J. Redmon and A. Farhad **“YOLOv3: An Incremental Improvement”**
Yield estimation method based on the study **"Estimating agricultural yields ”** by Prof. Bogdan Kulig, PhD, Agricultural University of Cracow.

This software is provided under license **MIT 2023**.
[MIT](https://choosealicense.com/licenses/mit/)

## Specific Requirements

| Index | Description                                                                         | Priority   | Category          |
|-------|-------------------------------------------------------------------------------------|------------|-------------------|
| W1    | The program must be easy to use and install                                          | required   | non-functional    |
| W2    | The program should include a user manual                                             | required   | non-functional    |
| W3    | The software should be capable of reading any image file                             | required   | functional        |
| W4    | The software should indicate the location of wheat ears in the image                 | required   | functional        |
| W5    | The model must be able to distinguish wheat ears among other plants                  | required   | functional        |
| W6    | The model should detect at least 95% of all wheat ears                               | required   | functional        |
| W7    | The software must allow running the model using a GPU                                | required   | functional        |
| W8    | The software must read the input values for the yield function                       | required   | functional        |
| W9    | The software must correctly calculate the estimated yield values                     | required   | functional        |
| P1    | The program allows saving an image with detection results to disk storage            | useful     | functional        |
| P2    | The program adjusts the interface size to the user's monitor dimensions              | useful     | functional        |
| P3    | To improve readability, the user can change the bounding box color                   | useful     | functional        |
| O1    | The software automatically performs prediction after loading an image file           | optional   | functional        |
| O2    | The software should perform detection in no more than 3 seconds                      | optional   | non-functional    |
| O3    | The interface adjusts the language of the software to the country where it is used   | optional   | functional        |
| O4    | The interface language is selected from a list                                       | optional   | functional        |
| O5    | The image file can be loaded directly from a network location                        | optional   | functional        |
| O6    | The application should perform detection on an image directly from the device camera | optional   | functional        |


## Directory structure

```{bash}
├── main.py
├── LICENSE.md
├── README.md
├── requirements.txt
├── src
│   ├── detect.py
│   ├── model_f.py
├── static
│   ├── css
│   │   └── styles.css
│   ├── favicon.ico
│   ├── js
│   │   └── utils.js
│   └── temp
└── templates
    └── index.html
```

### Development and Runtime Architecture
- **Version control**: Git 2.17.1  
- **IDE**: Jupyter Lab 3.1.7  
- **Python environment**: Anaconda 4.10.3  
- **Software language**: Python 3.7.11  
- **Framework**: Flask 1.1.2  
- **Deep Learning**: TensorFlow 1.13.1  
- **Image processing**: OpenCV 4.5.2.52, Albumentations 1.0.3  
- **Data analysis**: NumPy 1.21.5, scikit-learn 1.1.3  
- **Visualization**: Matplotlib 3.4.2  
- **Monitoring of model training processes**: TensorBoard 2.10.1  
- **Graphics processor support environment**: CUDA 11.4  

The software was developed and tested on a virtual machine with **Ubuntu 18.04.6 LTS**

## Requirements Testing

**Requirement W1**  
The launch consists of three steps, from downloading the repository to running the command line:

1. **Download the repository**  
   The project can be downloaded from the repository using the command:
   ```bash
   git clone https://github.com/DominikMAI/WheatDetectionModel.git
   cd WheatDetectionModel
   ```

2. **Download the detection model weights**  
   The model weights are needed for the model to work correctly. To download the weights, the curl tool is required.  
   The weights from the training can be downloaded from a network location:
   ```bash
   cd weights
   curl -o weights.h5 -L 'https://drive.google.com/uc?export=download&confirm=yes&id=1t9_0HlgSjF9UpboXfd2sJsuN2anM7Baq'

   python main.py --weights_dir=./weights/yolov5.hdf5
   ```

3. **Install the required libraries using the Anaconda command:**
   ```bash
   conda install --file requirements.txt
   ```

4. **Run the application interface (with default configuration):**
   ```bash
   python main.py
   ```

**Requirement W2**  
1. To view available arguments, run the program with the `--help` flag. Using the help argument allows you to see the available functions and the default values of the arguments:
   ```bash
   python main.py --help
   ```

**Requirement W3**  
To read the input image:

1. **Run the application:**
   ```bash
   python main.py
   ```
2. In the web browser address bar, enter the address `127.0.0.1:5000` and press Enter.
3. In the running application, click the "Browse" button, select a field image from the local disk in the opened dialog window.
4. After loading, the image should be displayed in the application's graphical interface.

**Requirement W4**  
1. Load the field image into the application interface.
2. Then click "Prediction".
3. After about 2 seconds, the displayed image in the interface should be replaced with an image showing the locations of wheat ears.

**Requirement W5**  
Verification of the requirement includes the following steps:

1. After clicking "Prediction," check whether the model correctly indicates the location of most wheat ears.
2. Among the detection results, there should not be any plants other than wheat.

**Requirement W6**  
1. Run the application.
2. Perform detection on an image located in the test data directory.

**Requirement W7**  
1. To use the GPU for prediction, the `gpu` argument must be placed in the command line.  
   GPU usage: none - `""`, one device - `"0"`, two devices - `"0,1"`, etc.
   ```bash
   python main.py --gpu='0'
   ```

**Requirement W8**  
1. To input values, enter them in the fields in order:  
   - **Average number of grains per ear**: During fieldwork, note the average number of grains per ear.  
   - **1000 grain mass**: Mass of 1000 grains in grams (g).  
   - **Extent of damage**: Previously noted yield loss. The loss should be given in percentage points.  

**Requirement W9**  
1. After entering the values, click the "Calculate Yield" button.
2. After clicking the button, the estimated yield value should immediately appear, along with the full message, e.g., "The yield is: 27.086 dt/ha."  
   The estimated yield is given in dt/ha.  
   Different ranges of values are accepted for different wheat species:

| Species         | Number of ears per 1m2 | Number of grains per ear | 1000 Grain Weight (MTN) | Grain Yield (dt/ha) |
|-----------------|------------------------|--------------------------|-------------------------|---------------------|
| Winter wheat    | 450-750                | 32.2-50.1                | 41.7-52.9               | 64.5-104.1          |
| Spring wheat    | 480-750                | 21.8-31.8                | 35.7-47.1               | 46.5-83.0           |
| Durum wheat     | 520-635                | 16.5-22.5                | 48.9-54.5               | 42.5-70.2           |

**Requirement P1**  
1. Perform the detection described above.
2. In the `./temp` directory, there should be a saved graphic file with the detection results.

**Requirement P2**  
1. After launching the application (see Requirement W2) and loading the image, the interface should fully fit within the monitor area.  
   All interface elements should be visible to the user.

**Requirement P3**  
1. To configure the color of the detection bounding boxes, use the `--bbox_color` argument in the command:
   ```bash
   python main.py --bbox_color='darkred'
   ```
   A source of available colors can be found at: [Matplotlib Named Colors](https://matplotlib.org/3.1.1/_images/sphx_glr_named_colors_003.png).  
   This option has been introduced because different images may have different brightness levels, and the default bounding box colors may not be visible.

