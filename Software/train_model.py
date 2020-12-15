#!/usr/bin/env python
"""
Steering angle prediction model

jay.urbain@gmail.com

"""
import cv2
import csv
import numpy as np
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, ELU
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
import time
import sklearn

import matplotlib.pyplot as plt


# basic net
def getModel(model="basicModel"):
    if model == "basicModel":
        return basicModel()
    elif model == "nVidiaModel":
        return nVidiaModel()
    elif model == "nVidiaModelRegularization":
        return nVidiaModelRegularization()
    elif model == "commaAiModel":
        return commaAiModel()
    elif model == "commaAiModelPrime":
        return commaAiModelPrime()

def basicModel(time_len=1):
    """
    Creates basic single layer autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten(input_shape=(160,230,3)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def preprocess(image):  # preprocess image
    import tensorflow as tf
    return tf.image.resize_images(image, (200, 66))
def newNvidia():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(preprocess))
    model.add(Lambda(lambda x: (x/ 127.0 - 1.0)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    #model.add(Dense(units=1164, activation='relu'))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    print(model.summary())
def nVidiaModel():
    """
    Creates nVidia autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127) - 1, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,(5,5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36,(5,5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48,(5,5),strides=(2, 2),  activation='relu'))
    model.add(Convolution2D(64,(3,3), activation='relu'))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def nVidiaModelRegularization():
    """
    Creates nVidia autonomous car  model
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001),dim_ordering='tf'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001),dim_ordering='tf'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001),dim_ordering='tf'))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def commaAiModel(time_len=1):
    """
    Creates comma.ai autonomous car  model
    Reduce dropout
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def commaAiModelPrime(time_len=1):
    """
    Creates comma.ai enhanced autonomous car  model
    Replace dropout with regularization
    Add 3 additional convolution layers
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    # Add three 5x5 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", W_regularizer=l2(0.001)))
    model.add(ELU())

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Flatten())

    # model.add(Dropout(.2))
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())

    # model.add(Dropout(0.50))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    return model


def getDrivingLogs(path, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def getImages(path):
    """
    Get all training images on the path `dataPath`.
    Returns `([centerPaths], [leftPath], [rightPath], [measurement])`
    """
    directories = [x[0] for x in os.walk(path)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centerTotal = []
    leftTotal = []
    rightTotal = []
    measurementTotal = []
    for directory in dataDirectories:
        lines = getDrivingLogs(directory, skipHeader=True)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(  line[0].strip())
            left.append( line[1].strip())
            right.append( line[2].strip())
        centerTotal.extend(center)
        leftTotal.extend(left)
        rightTotal.extend(right)
        measurementTotal.extend(measurements)

    return (centerTotal, leftTotal, rightTotal, measurementTotal)

def combineCenterLeftRightImages(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([imagePaths], [measurements])
    """
    imagePaths = []
    imagePaths.extend(center)
    imagePaths.extend(left)
    imagePaths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (imagePaths, measurements)

def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            try:
                for imagePath, measurement in batch_samples:
                    
                    originalImage = cv2.imread(imagePath)
                    # print(imagePath)
                    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
              
                    image = image[80:240,:,:]
                    images.append(image)
                    angles.append(measurement)
                    # Flipping
                    # images.append(cv2.flip(image,1))
                    # angles.append(measurement*-1.0)

                # trim image to only see section with road
                inputs = np.array(images)
                outputs = np.array(angles)
                yield sklearn.utils.shuffle(inputs, outputs)
            except Exception as e:
                print(e)
                print(imagePath)
import os

path = './'

files = os.listdir(path)
num_image = []
angle_data = []
angle_image = []
samples = []
for file_name in files:
    if(file_name.find(".") == -1):
        print(file_name)
        
        try:
            file_data = open( file_name + '.txt') 
            for data in file_data.readlines():
            
                X = data.split(':')
                num_image.append(X[0])
                # print(data)
                angle_image.append(float(X[1].replace("\n","")))
                samples.append( ("./" +file_name +"/" + X[0] +".jpg", float(X[1].replace("\n",""))/125.0) )
                # print(float(X[1].replace("\n",""))/108)
        except Exception as e:
            print(e)

# Reading images locations.

print('Total Images: {}'.format( len(samples)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Model creatio
# model = getModel(model="nVidiaModel")
# model = getModel(model="commaAiModel")
model = getModel(model="nVidiaModelRegularization")
#model = getModel(model="basicModel")
model.summary()


filepath="weights-improvement-{epoch:02d}.h5"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    save_weights_only=False,
    monitor="loss",
    mode="min",
    save_best_only=True,
 
)

# Train the model
callback = EarlyStopping(monitor='loss', patience=3)
history_object = model.fit_generator(train_generator, steps_per_epoch= \
                 len(train_samples)/64, validation_data=validation_generator, \
                 validation_steps=len(validation_samples)/64, nb_epoch=15, verbose=1,callbacks=[callback,model_checkpoint_callback])


# history = model.fit(train_samples[:,0],train_samples[:,1], np.zeros(5),epochs=10, batch_size=1, callbacks=[callback], verbose=0)

# model.save('model_nVidia.h5')
# model.save('model_commaAiModel_e10.h5')
model.save('nVidiaModelRegularization_e5.h5')
#model.save('model_commaAiModelPrime_e20.h5')
# model.save('model_commaAiModelPrime_e5.h5')

print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])



plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
