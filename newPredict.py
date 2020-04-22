import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import model_from_json

# Loading the pre-trained model(new from JSON)
json_file = open('model_cnn_final(h15).json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Best-weights-my_model-029-0.0226-0.9923.hdf15")
print("Loaded model from disk")

# Loading the pre-trained model(old)
# from keras.models import load_model
# model = load_model('model.h13')

# Compiling the loaded model
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
img_rows, img_cols = 200, 200
# Pre-processing test images
test_list = []
TESTDIR ='C:/Users/Tassi/PycharmProjects/thesis/Sentdex_data/Test'
images = glob.glob(os.path.join(TESTDIR, '*jpg'))

for img in images:
    filename = os.path.basename(img)
    print('Loading the images of dataset-'+'{}\n'.format(img))
    path = os.path.join(TESTDIR, img)
    input_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    input_img_resized = cv2.resize(input_img,(img_rows, img_cols))
    img_data = np.array(input_img_resized)
    img_data = img_data.astype('float32')
    img_data /= 255
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=3)
    test_list.append(img_data)

print(img_data.shape)
print(img_data[0].shape)

csvList = []
# Predicting the test images
for test_image in test_list:
    probabilities = (model.predict(test_image))
    print(probabilities)
    confidence = np.nanmax(probabilities) # nanmax returns the maximum value ina array ignoring the NAN value.
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
with open('newClassifier.csv', 'a') as csvFile:
     writer = csv.writer(csvFile)
     writer.writerows(csvList)
     csvFile.close()

# Visualizing the intermediate layer
def get_featuremaps(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations

layer_num=0
filter_num=0

activations = get_featuremaps(model, int(layer_num), test_image)

print (np.shape(activations))
feature_maps = activations[0][0]
print(np.shape(feature_maps))

if K.image_dim_ordering()=='th':
    feature_maps = np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print(feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:, :, filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
    ax = fig.add_subplot(subplot_num, subplot_num, i+1)
    ax.imshow(feature_maps[:,:,i],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')