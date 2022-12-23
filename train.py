import cv2
import numpy as np
import tensorflow.keras.models
import keras.layers 

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
 


train_dir = 'data/train'
arr=np.array(train_dir)
print(arr.shape)

val_dir = 'data/test'
tarin_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = tarin_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


emotion_model = tensorflow.keras.models.Sequential() 
emotion_model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
emotion_model.add(keras.layers.Dropout(0.25))
emotion_model.add(keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
emotion_model.add(keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
emotion_model.add(keras.layers.Dropout(0.25))
emotion_model.add(keras.layers.Flatten())
emotion_model.add(keras.layers.Dense(1024,activation='relu'))
emotion_model.add(keras.layers.Dropout(0.25))
emotion_model.add(keras.layers.Dense(7,activation='softmax'))


emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])

emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50, 
        validation_data=validation_generator,
        validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')

