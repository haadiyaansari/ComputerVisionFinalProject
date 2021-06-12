#https://www.mydatahack.com/building-alexnet-with-keras/

from keras import layers
from keras import preprocessing
from keras import models
from keras import optimizers
import numpy as np

# Initialize and create the CNN

TRAINING_PATH = 'chest_xray/train'
TESTING_PATH = 'chest_xray/test'
TARGET_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 3)

classifier = models.Sequential()
classifier.add(layers.Convolution2D(96, (11, 11), strides = 4, input_shape = INPUT_SHAPE, activation = 'relu'))
classifier.add(layers.BatchNormalization())
classifier.add(layers.MaxPooling2D(pool_size = (3,3), strides = 2))
classifier.add(layers.Convolution2D(256, (5, 5), padding = 'same', activation = 'relu'))
classifier.add(layers.BatchNormalization())
classifier.add(layers.MaxPooling2D(pool_size = (3,3), strides = 2))
classifier.add(layers.Convolution2D(384, (3, 3), activation = 'relu'))
classifier.add(layers.Convolution2D(384, (3, 3), activation = 'relu'))
classifier.add(layers.Convolution2D(256, (3, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (3,3), strides = 2))

classifier.add(layers.Flatten())

classifier.add(layers.Dense(4096, activation = 'relu'))
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(4096, activation = 'relu'))
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(1000, activation = 'relu'))
classifier.add(layers.Dense(1, activation = 'sigmoid'))

# Compile classifier
# Learning rates of 0.0001, 0.001, and 0.01 were used for the learning rate experiements
opt = optimizers.Adam(learning_rate=.01)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting CNN to the images
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(TRAINING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(TESTING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
classifier.fit_generator(training_set, steps_per_epoch=5216/32, epochs=25, validation_data=test_set, validation_steps = 624/32)

