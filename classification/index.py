# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2

from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout
from keras.models import Sequential
from keras import regularizers
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
    shape = (40, 40)

    for filename in os.listdir(train_path):
        name, res = filename.split(".")[0], filename.split(".")[1]
        if res == "jpg":
            img = cv2.imread(os.path.join(train_path, filename))
            name = name.split("_")[0]
            if name not in labels_map:
                continue
            train_labels.append(labels_map[name])
            img = cv2.resize(img, shape)
            train_images.append(img)

    #train_labels = pd.get_dummies(train_labels).values
    train_labels = np.array(train_labels)
    train_images = np.array([x.flatten() for x in train_images])
    train_images = train_images / 255.0

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
        Dense(3, activation="linear")
    ], name="model1")

    model2 = Sequential([
        Dense(40, activation="relu"),
        Dense(30, activation="relu"),
        Dense(20, activation="relu"),
        Dense(3, activation="linear")
    ], name="model2")

    model3 = Sequential([
        Dense(70, activation="relu"),
        Dense(60, activation="relu"),
        Dense(45, activation="relu"),
        Dense(20, activation="relu"),
        Dense(3, activation="linear")
    ], name="model3")



    lambdas = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    for lambda_ in lambdas:
        yield Sequential([
            Dense(70, activation="relu", kernel_regularizer=regularizers.l2(lambda_)),
            Dense(60, activation="relu", kernel_regularizer=regularizers.l2(lambda_)),
            Dense(45, activation="relu", kernel_regularizer=regularizers.l2(lambda_)),
            Dense(20, activation="relu", kernel_regularizer=regularizers.l2(lambda_)),
            Dense(3, activation="linear")
        ], name=f"final_model_reg_{str(lambda_)}")

    model4 = Sequential([
        Dense(100, activation="relu"),
        Dense(150, activation="relu"),
        Dense(50, activation="relu"),
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(3, activation="linear")
    ], name="model4")


def run_model(model, x_train, y_train, x_val, y_val):
    print(f"training model {model.name}")
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    history = model.fit(x_train,y_train,epochs=50,batch_size=50, validation_data=(x_val,y_val), verbose=0)
    plot_los(history, model.name)
    loss_last = history.history["loss"][-2:]
    print(f"model train loss last 2 {loss_last}")
    vloss_last = history.history["val_loss"][-2:]
    print(f"model validation loss last 2  {vloss_last}")

    train_accuracy = calc_accuracy(x_train, y_train, model)
    print(f"model train_accuracy {train_accuracy}")

    val_accuracy = calc_accuracy(x_val, y_val, model)
    print(f"model val_accuracy {val_accuracy}")

    print(f"done training model {model.name}")
    print(f"_________________________________")


def run():
    labels = {
        "banana": 1, 
        "apple": 0,
        #"mixed": 2,
        "orange": 2, 
    }
    X, Y = load_data("classification/data/train", labels)
    x_train, x_val, y_train, y_val = train_test_split(X, Y, random_state=1)
 
    X, Y = load_data("classification/data/test", labels)
    x_train_more_data, x_test, y_train_more_data, y_test = train_test_split(X, Y, random_state=1)
    x_train_more_data = np.append(x_train_more_data, x_train, axis=0)
    y_train_more_data = np.append(y_train_more_data, y_train, axis=0)

    #x_val = np.append(x_val, x_test, axis=0)
    #y_val = np.append(y_val, y_test, axis=0)

    print(x_train.shape, y_train.shape)

    for i, model in enumerate(get_models()):
        run_model(model, x_train, y_train, x_val, y_val)
        #run_model(model, x_train_more_data, y_train_more_data, x_val, y_val)
        

def calc_accuracy(x, y, model):
    ythat = model.predict(x)
    ythat_p = tf.nn.softmax(ythat)
    ythat_cat = np.argmax(ythat_p, axis=1)

    report = classification_report(y, ythat_cat)
    print(report)

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

def plot_los(history, name):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'model {name} loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

