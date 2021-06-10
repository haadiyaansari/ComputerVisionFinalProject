#https://www.mydatahack.com/building-alexnet-with-keras/

from keras import layers
from keras import preprocessing
from keras import models
import matplotlib
import cv2
import numpy as np

# #Initialize and create the CNN
#
TRAINING_PATH = 'chest_xray/train'
TESTING_PATH = 'chest_xray/test'
TARGET_SIZE = (300, 300)
INPUT_SHAPE = (300, 300, 3)
EPOCHS = 2
#
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
#
classifier.add(layers.Flatten())
#
classifier.add(layers.Dense(4096, activation = 'relu'))
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(4096, activation = 'relu'))
classifier.add(layers.Dropout(0.5))
classifier.add(layers.Dense(1000, activation = 'relu'))
classifier.add(layers.Dense(1, activation = 'sigmoid'))

# #Compile classifier
classifier.compile(optimizer = 'SGD', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()

#
# #Fitting CNN to the images
train_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(TRAINING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(TESTING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='binary')
stats = classifier.fit(training_set, steps_per_epoch=5216/32, epochs=EPOCHS, validation_data=test_set, validation_steps = 624/32)

with open("stats.txt", 'w') as fp:
    for i in range(EPOCHS):
        fp.write(f'{stats.history["loss"][i]}\t{stats.history["accuracy"][i]}\t{stats.history["val_loss"][i]}\t{stats.history["val_accuracy"][i]}\n')


'''
for i in range(96):
    weights = classifier.layers[0].get_weights()[0][:,:,:,i]
    minimum, maximum = weights.min(), weights.max()
    weights = (np.absolute(weights) - minimum) / (maximum - minimum)
    cv2.imwrite(f'Layer1_Filter{i}.jpg', weights*255)
'''