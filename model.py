#KERAS
import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras import callbacks
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import cv2
from tqdm import tqdm
import pickle
import itertools
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

##data
PATH = 'C:/Users/Tassi/PycharmProjects/thesis/input_data'    # path of folder of images

#input image dimensions
img_rows, img_cols = 200, 200

batch_size = 35
nb_classes = 2
nb_epoch = 35

labels_name = {'attentive':0,'inattentive':1}
img_data_list = []
labels_list = []

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'attentive': return 0
    elif word_label == 'inattentive': return 1

for img in tqdm(os.listdir(PATH)):
    print('Loading the images of dataset-'+'{}\n'.format(img))
    label = label_img(img)
    path = os.path.join(PATH, img)
    input_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    input_img_resized=cv2.resize(input_img,(img_rows,img_cols))
    img_data_list.append(input_img_resized)
    labels_list.append(label)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

labels = np.array(labels_list)
# print the count of number of samples for different classes
print('No. of labels:', np.unique(labels, return_counts=True))

X = np.array(img_data).reshape(-1, img_rows, img_cols, 1)
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, nb_classes)

# pickle the data set
pickle_out = open("features.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("labels.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

# Load the pickled data
pickle_in = open("features.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("labels.pickle", "rb")
Y = pickle.load(pickle_in)

#Shuffle the dataset
x, y = shuffle(X, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=4)

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# CNN-model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# saving the logs for training
log_dir="logs\\attentive-vs-inattentive-CNN\\plugins\\profile2\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir)
callbacks_list=[tensorboard]
callbacks_list.append(EarlyStopping(monitor='val_loss', patience=17))

# Normal Training
# hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                 # verbose=1, validation_data=(X_test, y_test))

# Viewing model_configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
# plotting the model network
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Training with callbacks
# filename='model_trainwithlogs_new.csv'
# csv_log=callbacks.CSVLogger(filename, separator=',', append=False)
# early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=19, verbose=0, mode='min')
# filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf15"
# checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
# callbacks_list = [csv_log,early_stopping,checkpoint]
# hist = model.fit(X_train, y_train, batch_size=35, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)
# print(hist.epoch)

# saving weights
# fname = "final-CNN model-attentive vs inattentive.hdf15"
# model.save_weights(fname, overwrite=True)
# model.save('model.h15')

#For future use in json file
# model_json = model.to_json()
# with open("model_cnn_final(h15).json", "w") as json_file:
    # json_file.write(model_json)
# serialize weights to HDF format
# model.save_weights("model_weights.hdf15")

# Loading the pre-trained model(new from JSON)
from keras.models import model_from_json
json_file = open('model_cnn_final(h15).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Best-weights-my_model-029-0.0226-0.9923.hdf15")
print("Loaded model from disk")
import pandas as pd
log_data = pd.read_csv('model_trainwithlogs_new.csv', sep=',', engine='python')

# visualizing losses and accuracy
# plotting loss
train_loss = log_data['loss']
val_loss = log_data['val_loss']
train_acc = log_data['acc']
val_acc = log_data['val_acc']
xc = range(max(log_data['epoch']) + 1)
plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','validation'])
plt.style.use(['classic'])
plt.style.use(['bmh'])
plt.style.use(['ggplot'])
plt.show()
# plotting accuracy
plt.figure(2,figsize=(7,5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','validation'], loc=4)
plt.style.use(['classic'])
plt.style.use(['ggplot'])
plt.show()

# Evaluating model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])
test_image = X_test[0:1]
print(test_image.shape)

# Predicting images
print('Predicted Labels:', model.predict_classes(X_test[1:5]))  # Predicted Labels
print('Actual Labels:', y_test[1:5])        # Actual Labels
y_true = np.argmax(y_test, axis=1)
print('True Labels:', y_true)
print(y_true.shape)
# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = model.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
## OR
# y_pred = model.predict_classes(X_test)
# print(y_pred)

## Predicting Probabilities
p = model.predict_proba(X_test)
print('Again true: ', p)
p_value = p[:, 1]        # this slicing means taking all the rows but only 2nd column values
print('Again true value: ', p_value)
target_names = ['class 0(ATTENTIVE)', 'class 1(INATTENTIVE)']
print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))

# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
np.set_printoptions(precision=2)
plt.figure()

# Plot non-normalized confusion matrix
fig = plot_confusion_matrix(cnf_matrix, classes=target_names, title='Confusion matrix')
plt.figure()

# Plot normalized confusion matrix
# fig = plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True, title='Normalized confusion matrix')
#plt.figure()
plt.show()


# Plotting the ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_pred, y_true)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='magenta', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()