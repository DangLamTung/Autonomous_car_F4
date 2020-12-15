import os
import cv2
from time import sleep
from subprocess import call
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
path = './'
model_path = "./nVidiaModelRegularization_e5.h5"
import numpy as np
files = os.listdir(path)
num_image = []
angle_data = []
angle_image = []
samples = []

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480


from keras.models import load_model
import h5py

f = h5py.File(model_path, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
    print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)

model = load_model(model_path)


cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)


cv2.namedWindow("Steering Wheel1", cv2.WINDOW_NORMAL)
cv2.moveWindow("Steering Wheel1", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)

img = cv2.imread("wheel.jpg", 0)
rows,cols = img.shape
smoothed_angle = 0
smoothed_angle1 = 0
a_max = -100
a_min = 100
angle = 0
steering_angle = []
for file_name in files:
    if(file_name.find(".") == -1):
        print(file_name)
        
        try:
            file_data = open( file_name + '.txt') 
            for data in file_data.readlines():
            
                X = data.split(':')
                num_image.append(X[0])

                # print(data)
                
                if(a_max < float(X[1].replace("\n",""))):
                    a_max = float(X[1].replace("\n",""))
                if(a_min > float(X[1].replace("\n",""))):
                    a_min = float(X[1].replace("\n",""))
                samples.append( ("./" +file_name +"/" + X[0] +".jpg", float(X[1].replace("\n",""))) )
                imagePath = "./" +file_name +"/" + X[0] +".jpg"
                image = cv2.imread(imagePath)
                # print(imagePath)
                # image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                angle_image.append(float(X[1].replace("\n","")))
                # print(image.shape)
                image = image[80:240,:,:]
                angle = float(model.predict(image[None, :, :, :], batch_size=1))
                steering_angle.append(angle)
                print(angle  )
                cv2.imshow("", image)
  
               
                degrees = float(X[1].replace("\n","")) +0.0001
          
             

                # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                # and the predicted angle
                smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imshow("Steering Wheel", dst)

                smoothed_angle1 += 0.2 * pow(abs((angle*108 - smoothed_angle1)), 2.0 / 3.0) * (angle*108 - smoothed_angle1) / abs(angle*108 - smoothed_angle1)
                M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle1, 1)
                dst = cv2.warpAffine(img,M,(cols,rows))
                cv2.imshow("Steering Wheel1", dst)
                sleep(0.001)
                if cv2.waitKey(1) == ord('q'):
                    break
        except Exception as e:
            print(e)
steering_angle = np.array(steering_angle)
angle_image = np.array(angle_image)

print(((steering_angle - angle_image)).mean(axis=None))
print(a_max)
print(a_min)

