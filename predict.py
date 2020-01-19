# Import libraries
import os.path as path
import cv2
import glob
import numpy as np
import os

# Loading the pre-trained model
from keras.models import load_model
model = load_model('model.h7')
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# Loading and pre-processing new images
num_channel = 1
mypath = "C:/Users/Tasnia Tabassum/Desktop/test_inattentive"
images = glob.glob(path.join(mypath, '*jpg'))
csvList = []
for files in images:
    filename = os.path.basename(files)
    test_image = cv2.imread(files)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (200, 200))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    print(test_image.shape)
    if num_channel == 1:
        test_image = np.expand_dims(test_image, axis=3)
        test_image = np.expand_dims(test_image, axis=0)
        print(test_image.shape)

# Predicting the test images
        probabilities = (model.predict(test_image))
        print(probabilities)
        confidence = np.nanmax(probabilities)  # nanmax returns the maximum value ina array ignoring the NAN value.
        print(confidence)
        classes = np.argmax(probabilities)  # argmax returns the maximum valued index in an array
        # classes = model.predict_classes(test_image)
        print(classes)
        if classes == 1:
            prediction = 'inattentive'
        else:
            prediction = 'attentive'
    csvList.append([filename, confidence, classes, prediction])
print(len(csvList))
##append in csv
import csv
with open('classifier.csv', 'a') as csvFile:
     writer = csv.writer(csvFile)
     writer.writerows(csvList)
     csvFile.close()
