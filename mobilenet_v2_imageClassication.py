import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import cv2 as cv
import PIL.Image as Image

from tensorflow.python.keras import layers
from tensorflow.python.keras import Sequential

IMAGE_SHAPE = (224, 224)
classifier = Sequential([hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v2/TensorFlow2/035-128-classification/2", input_shape = IMAGE_SHAPE + (3,))])

goldfish_img = Image.open("/Users/arhan.sheth/Documents/Codes/DX/transferLearning/tfLearning/goldfish.png").resize(IMAGE_SHAPE)
goldfish_img = np.array(goldfish_img)/255.0
print(goldfish_img.shape)