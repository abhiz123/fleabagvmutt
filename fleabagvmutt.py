import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
import matplotlib.pyplot as plt

import os.path
from tensorflow.keras.models import load_model


train_labels = []
train_samples = []

for i in range(50):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

train_labels,train_samples = shuffle(train_labels,train_samples)

scaler = MinMaxScaler((0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

#SETTING UP DATA DONE  NOW CREATING NETWORK

if os.path.isfile("model/model1.h5") is False:
    model = Sequential([
    Dense(units = 16, input_shape = (1,), activation = "relu"),
    Dense(units = 32, activation = "relu"),
    Dense(units = 2, activation = "softmax")
    ])

    model.compile(optimizer = Adam(learning_rate = 0.0001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    model.fit(x = scaled_train_samples,y = train_labels, validation_split = 0.1, batch_size = 10,epochs = 30, shuffle = "True", verbose = 2)

else:
    model = load_model("model/model1.h5")

model.summary()
print(model.get_weights())


#BUILDING TEST SET

test_labels = []
test_samples = []

for i in range(10):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

test_labels,test_samples = shuffle(test_labels,test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

predictions = model.predict(x = scaled_test_samples, batch_size = 10, verbose = 0)

rounded_predictions = np.argmax(predictions, axis = -1)

# NOW TO PLOT CONFUSION MATRIX AND UNDERSTAND DATA

cm = confusion_matrix(y_true = test_labels, y_pred = rounded_predictions)

# A UNIVERSAL CONFUSION MATRIX PLOTTING FUNCTION(FORM SKLEARN WEBSITE)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm_plot_labels = ["no side effects","has side effects"]
# plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = "Confusion Matrix")


#SAVING AND LOADING models

if os.path.isfile("model/model1.h5") is False:
    model.save("model/model1.h5")    #to save entire model + weights

# model = load_model("model/model1.h5")  #to load model

#--------------------------------------------------------------------------------
#To Only save architecture without weights
# json_string = model.to_json()
# model = model_from_json(json_string)

#--------------------------------------------------------------------------------
#To Only store weights without architecture
# weights = model.save_weights("model/model1.h5")

#new_model = Sequential([
# Dense(units = 16, input_shape = (1,), activation = "relu"),
# Dense(units = 32, activation = "relu"),
# Dense(units = 2, activation = "softmax")
# ])

#new_model.load_weights("model/model1.h5")
