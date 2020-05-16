import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus


################## Model #####################################

classifier = Sequential()

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,1),activation='relu'))
classifier.add(MaxPool2D(2,2))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPool2D(2,2))

classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPool2D(2,2))

classifier.add(Flatten())

classifier.add(Dense(128,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(4,activation="softmax"))

classifier.compile(optimizer = 'adam',loss='categorical_crossentropy', metrics =['accuracy'])
################### Feeding the image data ##########################
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255,shear_range=0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip=True)
######################### Data Path #############################
train_path ='./Data/EmotionData2/Train'
test_path ='./Data/EmotionData2/Test'
train_set = train_datagen.flow_from_directory(train_path,
                                              target_size=(64, 64),
                                              color_mode='grayscale',
                                              class_mode='categorical')
validation_set = test_datagen.flow_from_directory(test_path,

                                                  target_size=(64, 64),
                                                  color_mode='grayscale',
                                                  class_mode='categorical'
                                                  )

classifier.fit_generator(train_set,
                        steps_per_epoch=1000,
                        epochs = 20,
                        validation_data = validation_set,
                        validation_steps =170,
                        shuffle = False)


classifier.save('emotion_detector_v2.h5')