# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2

from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

def load_data(train_path, labels_map):
    np.random.seed(1)

    train_images = []
    train_labels = []
    shape = (200, 200)

    for filename in os.listdir(train_path):
        name, res = filename.split(".")[0], filename.split(".")[1]
        if res == "jpg":
            img = cv2.imread(os.path.join(train_path, filename))
            name = name.split("_")[0]
            train_labels.append(labels_map[name])
            img = cv2.resize(img, shape)
            train_images.append(img)

    #train_labels = pd.get_dummies(train_labels).values
    train_labels = np.array(train_labels)
    train_images = np.array([x.flatten() for x in train_images])

    return train_images, train_labels

def print_random_data(X, y):   
    for i in range(10):
        index = np.random.choice(len(X))
        image = X[i].reshape(200, 200, 3)
        plt.imshow(image)
        plt.show()


def softmax(x):
    ez = np.exp(x)
    return ez / sum(ez)


def get_models():
    model1 = Sequential([
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(4, activation="linear")
    ])

    model2 = Sequential([
        Dense(40, activation="relu"),
        Dense(30, activation="relu"),
        Dense(20, activation="relu"),
        Dense(4, activation="linear")
    ])

    model3 = Sequential([
        Dense(70, activation="relu"),
        Dense(60, activation="relu"),
        Dense(45, activation="relu"),
        Dense(20, activation="relu"),
        Dense(4, activation="linear")
    ])

    model_ = Sequential([
        Dense(5000, activation="relu"),
        Dense(1000, activation="relu"),
        Dense(500, activation="relu"),
        Dense(200, activation="relu"),
        Dense(50, activation="relu"),
        Dense(20, activation="relu"),
        Dense(4, activation="linear")
    ])

    model4 = Sequential([
        Dense(100, activation="relu"),
        Dense(150, activation="relu"),
        Dense(50, activation="relu"),
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(4, activation="linear")
    ])

    return [model1, model2, model3, model4]


def run():
    labels = {
        "banana": 1, 
        "apple": 0,
        "mixed": 2,
        "orange": 3
    }
    X, Y = load_data("classification/data/train", labels)
    x_train, x_trainv, y_train, y_trainv = train_test_split(X, Y, random_state=1)
    X, Y = load_data("classification/data/test", labels)
    x_val, x_test, y_val, y_test = train_test_split(X, Y, random_state=1)
    print(x_train.shape, y_train.shape)

    for i, model in enumerate(get_models()):
        print(f"training model {i}")
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )
        history = model.fit(x_train,y_train,epochs=50,batch_size=50, validation_data=(x_val,y_val), verbose=0)
        plot_los(history)
        loss_last = history.history["loss"][-2:]
        print(f"model train loss last 2 {loss_last}")
        vloss_last = history.history["val_loss"][-2:]
        print(f"model validation loss last 2  {vloss_last}")

        train_accuracy = calc_accuracy(x_train, y_train, model)
        print(f"model train_accuracy {train_accuracy}")

        val_accuracy = calc_accuracy(x_val, y_val, model)
        print(f"model val_accuracy {val_accuracy}")

        print(f"done training model {i}")
        print(f"_________________________________")

def calc_accuracy(x, y, model):
    ythat = model.predict(x)
    ythat_p = tf.nn.softmax(ythat)
    ythat_cat = np.argmax(ythat_p, axis=1)
    accuracy = np.sum(y==ythat_cat)
    return accuracy / len(x)



def plot_accuracy1(history):

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_los(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

