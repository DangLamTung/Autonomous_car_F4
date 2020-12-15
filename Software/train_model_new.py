import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
import cv2
def generator(image_train, label_train, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    order = np.arange(len(image_train))
    print(order)
    while True:

        # Shuffle training data
        np.random.shuffle(order)
        x = image_train[order]
        y = label_train[order]
    

        for index in range(batch_size):
            x_train = image_train[index * batch_size:(index + 1) * batch_size]
            y_train = label_train[index * batch_size:(index + 1) * batch_size]
 
            yield (x_train), (y_train)
            # except Exception as e:
            #     print(e)
            #     print(imagePath)
import os

path = './'

files = os.listdir(path)
num_image = []
angle_data = []
angle_image = []
samples = []

image_train = []
label_train = []
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
                

                originalImage = cv2.imread("./" +file_name +"/" + X[0] +".jpg")
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                image = image[80:240,:,:] / 255.0 - 0.5


                image_train.append(image.astype(np.float32))
                label_train.append(float(X[1].replace("\n",""))/108)
                # print(float(X[1].replace("\n",""))/108)
        except Exception as e:
            print(e)
def val_generator(image_train, label_train, batch_size=32):
    """
    
    """


    while True:
        # We don't shuffle validation data
        for index in range(batch_size):
            x_val = image_train[index * batch_size:(index + 1) * batch_size]
            y1_val = label_train[index * batch_size:(index + 1) * batch_size]
          
            yield (x_val), (y1_val)
# Reading images locations.



# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(image_train,label_train, test_size=0.2)

print('Train samples: {}'.format(len(X_train)))
print('Validation samples: {}'.format(len(y_test)))

train_generator = generator(X_train,y_train, batch_size=32)
validation_generator = generator(X_test, y_test, batch_size=32)

train_samples = len(X_train)
validation_samples = len(y_test)
model = keras.Sequential(
    [
        
        keras.Input(shape=(160,320,3)),
    
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='linear'),
    ]
)

model.summary()

batch_size = 32
epochs = 100

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)

checkpoint_filepath = "/tmp/checkpoint"

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)
callbacks = [early_stopping_callback, model_checkpoint_callback]

model.compile(loss="mse", optimizer="adam", metrics=['MeanSquaredError'])

history_object = model.fit_generator(train_generator, steps_per_epoch= \
                 train_samples/batch_size, validation_data=validation_generator, \
                 validation_steps=validation_samples/batch_size, epochs=epochs, verbose=1,callbacks=callbacks)
# X_train = np.array(X_train)
# print(X_train.shape)
# history_object = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, verbose=1,callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])