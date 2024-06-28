import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image

from keras import layers, models, losses, optimizers
from keras import Sequential

# Pre-determining image input shape
IMAGE_SHAPE = (224, 224)

# Initializing classifier model with a proper TensorFlow Hub URL
classifier = models.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE + (3,))
    ])

# Initializing and pre-processing the sample input image
image_path = "/Users/arhan.sheth/Documents/Codes/DX/transferLearning/tfLearning/goldfish.png"
goldfish_img = Image.open(image_path).resize(IMAGE_SHAPE)
goldfish_img = np.array(goldfish_img) / 255.0

# Adding another dimension for prediction
goldfish_img = np.expand_dims(goldfish_img, axis=0)
res = classifier.predict(goldfish_img)

# Output the prediction result
predicted_index = np.argmax(res)
print(predicted_index)