#https://www.mydatahack.com/building-alexnet-with-keras/

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

#Initialize these varaibles before running 

TRAINING_PATH = ''
TESTING_PATH = ''
TARGET_SIZE = ''
INPUT_SHAPE = ''


#Initialize and create the CNN

classifier = Sequential()
classifier.add(Convolution2D(nb_filter = 96, nb_row=11, nb_col=11, strides = 4, input_shape = INPUT_SHAPE, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (3,3), strides = 2))
classifier.add(Convolution2D(nb_filter = 256, nb_row=5, nb_col=5, padding = 'same', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (3,3), strides = 2))
classifier.add(Convolution2D(nb_filter = 384, nb_row=3, nb_col=3, activation = 'relu'))
classifier.add(Convolution2D(nb_filter = 384, nb_row=3, nb_col=3, activation = 'relu'))
classifier.add(Convolution2D(nb_filter = 256, nb_row=3, nb_col=3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (3,3), strides = 2))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 4096, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim = 1000, activation = 'relu'))
classifier.add(Dense(output_dim = 7, activation = 'softmax'))

#Compile classifier
classifier.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(TRANING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory(TESTING_PATH, target_size=TARGET_SIZE, batch_size=32, class_mode='categorical')
classifier.fit_generator(training_set, steps_per_epoch=8000/32, epochs=25, validation_data=test_set, validation_steps = 2000/32)

