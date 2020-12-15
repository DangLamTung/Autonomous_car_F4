import sklearn
import cv2
import csv
import numpy as np
import os




def read_data(file_name):
    file_name = '2020-12-08 19-33-32'
    num_image = []
    angle_data = []
    angle_image = []
    samples = []
    file_data = open( '2020-12-08 19-33-32.txt') 
    for data in file_data.readlines():
        try:
            X = data.split(':')
            num_image.append(X[0])
            print(data)
            angle_image.append(float(X[1].replace("\n","")))
            samples.append( ("./" +file_name +"/" + X[0] +".jpg", float(X[1].replace("\n",""))) )
        except Exception as e:
            print(e)
        return samples
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
            print("a")
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)
                cv2.imshow("", image) 
  

                cv2.waitKey(0)  
            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            
            return sklearn.utils.shuffle(inputs, outputs)
# samples = read_data()
# print(samples)
# generator(samples, batch_size=32)

import os

path = './'

files = os.listdir(path)

for file_name in files:
    if(file_name.find(".") == -1):
        print(file_name)
        num_image = []
        angle_data = []
        angle_image = []
        samples = []
        file_data = open( file_name + '.txt') 
        for data in file_data.readlines():
            try:
                X = data.split(':')
                num_image.append(X[0])
                print(data)
                angle_image.append(float(X[1].replace("\n","")))
                samples.append( ("./" +file_name +"/" + X[0] +".jpg", float(X[1].replace("\n",""))) )
            except Exception as e:
                print(e)
