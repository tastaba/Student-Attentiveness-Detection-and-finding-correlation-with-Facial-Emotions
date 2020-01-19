#KERAS
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

##data
path1 = 'C:/Users/Tasnia Tabassum/PycharmProjects/thesis/input_data'    #path of folder of images
path2 = 'C:/Users/Tasnia Tabassum/PycharmProjects/thesis/input_data_resized'  #path of folder to save images

#path of folder to save images
listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)

for file in listing:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    #need to do some more processing here
    gray.save(path2 +'\\' +  file, "JPEG")

imlist = os.listdir(path2)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


im1 = array(Image.open('input_data_resized' + '\\'+ imlist[0]))  # open one image to get size
m,n = im1.shape[0:2]  #get the size of the images
imnbr = len(imlist)  #get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open('input_data_resized' + '\\' + im2)).flatten()
                  for im2 in imlist], 'f')

label = np.ones((num_samples,), dtype=int)
label[0:1750] = 0 #labeling the first 1750 images as 'attentive'
label[1750:] = 1 #labeling the last 1750 images as 'inattentive'

data, Label = shuffle(immatrix, label, random_state=2)  #shuffling the dataset
train_data = [data, Label]

print(train_data[0].shape)
print(train_data[1].shape)

#batch_size to train
batch_size = 25
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 17

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])
# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices(one hot encoding)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


#CNN-model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])

hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 verbose=1, validation_data=(X_test, Y_test))

# hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                 #verbose=1, validation_split=0.2)


# visualizing losses and accuracy
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(nb_epoch)
plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','validation'])
# print(plt.style.available) # use bmh, classic, ggplot for big pictures
plt.style.use(['classic'])
plt.show()
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','validation'],loc=4)
plt.style.use(['classic'])
plt.show()

# Predicting images/Evaluating model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))  # Predicted Labels
print(Y_test[1:5])  # Actual Labels

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
##OR
y_pred = model.predict_classes(X_test)
print(y_pred)

##Predicting Probabilities
p = model.predict_proba(X_test)
target_names = ['class 0(ATTENTIVE)', 'class 1(INATTENTIVE)']
print(classification_report(np.argmax(Y_test, axis=1), y_pred, target_names=target_names))
print(confusion_matrix(np.argmax(Y_test, axis=1), y_pred))

# saving weights
fname = "weights-Test-CNN.hdf7"
model.save_weights(fname, overwrite=True)
model.save('model.h7')
