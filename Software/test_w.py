import numpy as np
import cv2
import h5py
import os
from keras import __version__ as keras_version
path = './'
model_path = "./nVidiaModelRegularization_e1.h5"
import numpy as np
files = os.listdir(path)
from keras.models import load_model
import h5py

f = h5py.File(model_path, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')

if model_version != keras_version:
    print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)

model = load_model(model_path)
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame,(320,240))
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame

    image = frame [80:240,:,:]
    angle = float(model.predict(image[None, :, :, :], batch_size=1))
    print(angle*108)
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()