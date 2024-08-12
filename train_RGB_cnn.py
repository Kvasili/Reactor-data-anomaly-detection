'''
    USAGE
    python train_RGB_cnn.py


    references
    # https://youtu.be/9GzfUzJeyi0



    USAGE
    python train_RGB_cnn.py --model custom_GASF_v7.model


'''

from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import argparse
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime


print('\n######################## START ####################################\n')
start = datetime.datetime.now()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-p1", "--path1", required=True, help="path for the intial data")
# ap.add_argument("-p2", "--path2", required=True, help="path for the training data")
ap.add_argument("-m", "--model", required=True,
                help="model name to be save to disk")
args = vars(ap.parse_args())
print((args))


SIZE = 64
EPOCHS = 10

# TRAINING IMAGES
train_images = []
train_labels = []
path_for_training_images = "./SCRAM_training_GASF_small/training/*"
for directory_path in glob.glob(path_for_training_images):

    # label = directory_path.split("/")[-1]
    label = os.path.normpath(directory_path).replace(
        os.sep, '/').split('/')[-1]
    # label = label.split('\')[-1]

    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
# print("training labels:")
# print(train_labels)


# VALIDATION IMAGES
validation_images = []
validation_labels = []

path_for_validation_images = "./SCRAM_training_GASF_small/validation/*"
for directory_path in glob.glob(path_for_validation_images):
    fruit_label = os.path.normpath(directory_path).replace(
        os.sep, '/').split('/')[-1]
    # print("validation labels")
    print(fruit_label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        validation_images.append(img)
        validation_labels.append(fruit_label)

validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)
# print("validation labels:")
# print(validation_labels)


#####################################################################
###### This is for testing after the training  ######################

test_images = []
test_labels = []

path_for_validation_images = "./SCRAM_training_GASF_small/testing/*"
for directory_path in glob.glob(path_for_validation_images):
    fruit_label = os.path.normpath(directory_path).replace(
        os.sep, '/').split('/')[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        # print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


# Encode labels from text to integers.
le = preprocessing.LabelEncoder()
le.fit(test_labels)
# test_labels_encoded = le.transform(test_labels)

x_test = test_images / 255.0
# x_test = x_test / 255.0
# y_test_one_hot = to_categorical(y_test)

##########################################################
##########################################################


# Encode labels from text to integers.
le.fit(validation_labels)
validation_labels_encoded = le.transform(validation_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_val, y_val = train_images, train_labels_encoded, validation_images, validation_labels_encoded
# print(x_train, y_train, x_test, y_test)

# ###################################################################
# # Normalize pixel values to between 0 and 1
x_train, x_val = x_train / 255.0, x_val / 255.0

# #One hot encode y values for neural network.
y_train_one_hot = to_categorical(y_train)
# print("ytrain one hot: ")
# print(y_train_one_hot)
y_val_one_hot = to_categorical(y_val)
# print("yval one hot: ")
# print(y_val_one_hot)

#############################

activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation,
                      padding='same', input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation=activation,
                      padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(64, 3, activation=activation,
                      padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(64, 3, activation=activation,
                      padding='same', kernel_initializer='he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

# Add layers for deep learning prediction
x = feature_extractor.output
x = Dense(128, activation=activation, kernel_initializer='he_uniform')(x)
prediction_layer = Dense(2, activation='sigmoid')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

print("Model summary: ")
# print(cnn_model.summary())

##########################################
# Train the CNN model
# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=7)

history = cnn_model.fit(x_train, y_train_one_hot, epochs=EPOCHS,
                        validation_data=(x_val, y_val_one_hot), callbacks=[es])

##########################################
# save the CNN model
# define the path to the serialized output model after training
MODEL_PATH = args['model']
# cnn_model.save(MODEL_PATH, save_format="h5")

################################################
# plot the training and validation accuracy and loss at each epoch
# with the following parameters

plot_properties = {
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'axes.facecolor': 'white',  # Set background color to white
}

plt.rcParams.update(plot_properties)

# Create your plot
# x = np.arange(0, 1.1, 0.1)  # x values from 0 to 1 with increment of 0.1
# y = np.random.randint(1, 10, size=len(x))  # some sample y values

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training')
plt.plot(epochs, val_loss, 'r', label="Validation")
plt.title('Loss Custom Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.title('Accuracy custom network')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()


##############################################################################
###### This part of the code for extracting the indexes ######################

prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)
# print(prediction_NN)

# Confusion Matrix - verify accuracy of each class
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
# sns.heatmap(cm, annot=True)
# plt.show()

# show a nicely formatted classification report
print(classification_report(test_labels, prediction_NN))


end = datetime.datetime.now()
print("Training completed after {} ".format(end-start))
