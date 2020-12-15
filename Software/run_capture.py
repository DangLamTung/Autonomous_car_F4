import tensorflow as tf
import scipy.misc
import cv2
from subprocess import call
import time
# from vidgear.gears import CamGear
import serial
from time import sleep
import os
import h5py
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

# class ReadLine:
#     def __init__(self, s):
#         self.buf = bytearray()
#         self.s = s
    
#     def readline(self):
#         i = self.buf.find(b"\n")
#         if i >= 0:
#             r = self.buf[:i+1]
#             self.buf = self.buf[i+1:]
#             return r
#         while True:
#             i = max(1, min(2048, self.s.in_waiting))
#             data = self.s.read(i)
#             i = data.find(b"\n")
#             if i >= 0:
#                 r = self.buf + data[:i+1]
#                 self.buf[0:] = data[i+1:]
#                 return r
#             else:
#                 self.buf.extend(data)
# ser = serial.Serial(            
#      port='/dev/ttyUSB0',
#      baudrate = 115200,
#      parity=serial.PARITY_NONE,
#      stopbits=serial.STOPBITS_ONE,
#      bytesize=serial.EIGHTBITS,
#      timeout=1
# )

# rl = ReadLine(ser)
# def set_angle(set_point, velocity):
#     global ser
#     try: 
#         values = []
#         sleep(0.05)
#         data = []
#         data1 = []
#         start = 's'
#         end = 'e'
#         CRC = ""
        
#         esc1 = bin(set_point + 180)
#         esc2 = bin(velocity + 500)
#         esc3 = bin(1000)
#         esc4 = bin(1000)
    
#         esc = '0b00000000000'
#         esc = esc[2:(len(esc)-len(esc1)+2)]  + esc1[2:len(esc1)]
#         # print(len(esc))   
#         b1 = esc
#         #value 2
#         esc = '0b00000000000'
#         esc = esc[2:(len(esc)-len(esc2)+2)]  + esc2[2:len(esc2)]
#         b2 = esc
#         #value 3
#         esc = '0b00000000000'
#         esc = esc[2:(len(esc)-len(esc3)+2)]  + esc3[2:len(esc3)]
#         b3 = esc
        
#         esc = '0b00000000000'
#         esc = esc[2:(len(esc)-len(esc4)+2)]  + esc4[2:len(esc4)]
#         b4 = esc 

#         frame_check = (set_point + velocity + 1000 + 1000 + 500 + 180) %37
#         data_frame = b1 + b2 + b3 + b4

        
#         frame1 = '0b' + data_frame[0:8]
#         frame2 = '0b' + data_frame[8:16]
#         frame3 = '0b' + data_frame[16:24]
#         frame4 = '0b' + data_frame[24:32]
#         frame5 = '0b' + data_frame[32:40]
#         frame6 = '0b' + data_frame[40:44] + '0000'
        


#         data.append(int(frame1,2))
#         data.append(int(frame2,2))
#         data.append(int(frame3,2))
#         data.append(int(frame4,2))
#         data.append(int(frame5,2))
#         data.append(int(frame6,2))
#         data.append(frame_check)
#         # print(data)
#         ser.write(b's')
#         for i in data:
#             ser.write(bytes([i]))
            
#         ser.write(b'e') 
#         #    
       
#     except Exception as e: 
#         print(e)
        
FLAGS = tf.app.flags.FLAGS

"""model from nvidia's training"""
tf.app.flags.DEFINE_string(
    'model', './nvidia/model.ckpt',
    """Path to the model parameter file.""")
# generated model after training
# tf.app.flags.DEFINE_string(
#     'model', './data/models/model.ckpt',
#     """Path to the model parameter file.""")

tf.app.flags.DEFINE_string(
    'steer_image', './data/.logo/steering_wheel_image.jpg',
    """Steering wheel image to show corresponding steering wheel angle.""")

WIN_MARGIN_LEFT = 240
WIN_MARGIN_TOP = 240
WIN_MARGIN_BETWEEN = 180
WIN_WIDTH = 480

# define suitable tweak parameters for your stream.
options = {"CAP_PROP_FRAME_WIDTH":320, "CAP_PROP_FRAME_HEIGHT":240, "CAP_PROP_FPS":30}

# To open live video stream on webcam at first index(i.e. 0) device and apply source tweak parameters
# stream = cv2.VideoCapture(-1)


               
if __name__ == '__main__':
    #img = cv2.imread(FLAGS.steer_image, 0)
    #rows,cols = img.shape

    cap = cv2.VideoCapture(0)

    # Visualization init
   # cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
   # cv2.moveWindow("Steering Wheel", WIN_MARGIN_LEFT, WIN_MARGIN_TOP)
    cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
   # cv2.moveWindow("Capture", WIN_MARGIN_LEFT+cols+WIN_MARGIN_BETWEEN, WIN_MARGIN_TOP)

    
    smoothed_angle = 0
    i = 0

        # construct model
    
    try:       
        while(True):
            start = time.time()
            _,frame = cap.read()
            frame = cv2.resize(frame,(320,320))
            image = frame/ 255.0 - 0.5
            image = image[80:240,:,:]
            angle = float(model.predict(image[None, :, :, :], batch_size=1))
            
                    
            call("clear")
            print(angle*108*10 )
                    
            cv2.imshow("Capture", image)
            end = time.time()
            print("FPS: " + str(1/(end - start)))
                    #print("Captured image size: {} x {}").format(frame.shape[0], frame.shape[1])

                    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
                    # and the predicted angle
                #  smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
                # M = cv2.getRotationMatrix2D((cols/2,rows/2), -smoothed_angle, 1)
                #  dst = cv2.warpAffine(img,M,(cols,rows))
                # cv2.imshow("Steering Wheel", dst)

                    #i += 1
    except Exception as e:
        print(e)
cap.release()
cv2.destroyAllWindows()

