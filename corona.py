# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:05:51 2020

@author: kushagar
"""

import keras  #using Tensorflow backend for keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten, BatchNormalization
from keras.layers import SeparableConv2D,Conv2D


classifier = Sequential()  #making sequential classifier

#First block
classifier.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation = 'relu')) 
classifier.add(Conv2D(filters=32, kernel_size=(3,3),  activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Second block
classifier.add(SeparableConv2D(filters=64, kernel_size=(3,3), activation='relu'))
classifier.add(SeparableConv2D(filters=64, kernel_size=(3,3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
classifier.add(Flatten())

classifier.add(Dense(output_dim=256, activation='relu')) #setting up the first input layer
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid')) #Setting up the output layer
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']) #Compiing the model


# It is to import the user defined dataset(read more from keras documentation)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (64,64),
                                                 batch_size = 16,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size = (64,64),
                                            batch_size = 16,
                                            class_mode = 'binary')

#Fitting the classifier could take time depending on hardware of PC
classifier.fit_generator(training_set,
                         samples_per_epoch = 261,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 41)


#This is to save model for later use(optional)
classifier.save('corona.model')

classifier.summary()
