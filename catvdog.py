import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import os
import shutil
import random
import glob
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
import sys

from plotstuff import plot_confusion_matrix

model_name = ""
if(len(sys.argv) > 1):
    model_name = sys.argv[1]

all_images_path = "/Users/abhiz123/Desktop/dogs-vs-cats/data"
vgg16_model_path = "/Users/abhiz123/Desktop/fleabagvmutt/model/vgg16cd.h5"
basic_model_path = "/Users/abhiz123/Desktop/fleabagvmutt/model/basic.h5"
vgg16_downloaded_weights_path = "/Users/abhiz123/Desktop/fleabagvmutt/vgg16_downloaded_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5"


# Setting up our file structure
os.chdir(all_images_path)
if os.path.isdir("train/dog") is False:
    os.makedirs("train/dog")
    os.makedirs("train/cat")
    os.makedirs("test/dog")
    os.makedirs("test/cat")
    os.makedirs("validation/dog")
    os.makedirs("validation/cat")

    for c in random.sample(glob.glob("cat*"),500):
        shutil.move(c,"train/cat")
    for c in random.sample(glob.glob("dog*"),500):
        shutil.move(c,"train/dog")
    for c in random.sample(glob.glob("cat*"),100):
        shutil.move(c,"validation/cat")
    for c in random.sample(glob.glob("dog*"),100):
        shutil.move(c,"validation/dog")
    for c in random.sample(glob.glob("cat*"),50):
        shutil.move(c,"test/cat")
    for c in random.sample(glob.glob("dog*"),50):
        shutil.move(c,"test/dog")

train_path = "train"
validation_path = "validation"
test_path = "testv"

# Preprocessing of images
train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = train_path, target_size = (224,224), classes = ["cat","dog"], batch_size = 10)

validation_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = validation_path, target_size = (224,224), classes = ["cat","dog"], batch_size = 10)

test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory = test_path, target_size = (224,224), classes = ["cat","dog"], batch_size = 10, shuffle = False)
# imgs,labels = next(test_batches)



# CREATING OR LOADING A PREEXISTING CNN model
# os.chdir(saved_models_path)

if model_name == "basic":
    if os.path.isfile(basic_model_path) is False:
        model = create_basic_model()
    else:
        print("LOADING BASIC MODEL")
        model = load_model(basic_model_path)
else:
    if os.path.isfile(vgg16_model_path) is False:
        model = create_vgg16_model()
    else:
        model = load_model(vgg16_model_path)


model.summary()

#NOW FOR PREDICTION

# test_images, test_labels = next(test_batches)
predictions = model.predict(x = test_batches, verbose = 0)
for i in range (0,len(predictions)):
    print(i ,predictions[i])
cm = confusion_matrix(y_true = test_batches.classes, y_pred = np.argmax(predictions, axis = -1))


cm_plot_labels = ["cat","dog"]
plot_confusion_matrix(cm, classes = cm_plot_labels, title = "Confusion Matrix")

def create_basic_model():
    model = Sequential([
    Conv2D(filters = 32, kernel_size = (3,3), activation = "relu", padding = "same", input_shape = (224,224,3)),
    MaxPool2D(pool_size = (2,2), strides = 2),
    Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", padding = "same"),
    MaxPool2D(pool_size = (2,2), strides = 2),
    Flatten(),
    Dense(units = 2, activation = "softmax")
    ])
    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.fit(x = train_batches,validation_data = validation_batches, epochs = 10, verbose = 2)
    model.save("basic.h5")
    return model


def create_vgg16_model():
    vgg16_model = VGG16(weights = vgg16_downloaded_weights_path)

    model = Sequential()
    for layer in vgg16_model.layers[:-1]:
        model.add(layer)
    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(units = 2, activation = "softmax"))
    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    model.fit(x = train_batches,validation_data = validation_batches, epochs = 5, verbose = 2)
    model.save("vgg16cd.h5")

    return model
