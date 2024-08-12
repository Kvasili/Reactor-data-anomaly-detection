
'''
    code for training binary classification for RGB images using RESNET weights pretrained on imagenet



    USAGE
    python train_resnet.py --path ./SCRAM_training_GASF_small --model resnet_GASF_v5_small.model

'''

# import the necessary packages
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
import os
import argparse
from sklearn.metrics import confusion_matrix


print('\n######################## START ####################################\n')
start = datetime.datetime.now()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path for training images")
ap.add_argument("-m", "--model", required=True, help="saved model name")
args = vars(ap.parse_args())
print((args))

MODEL_NAME = args['model']
CLASSES = ["Abnormal", "Normal"]

# set the matplotlib backend so figures can be saved in the background
# matplotlib.use("Agg")


BASE_PATH = args['path']
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])


INIT_LR = 0.0001
batch_size = 32
NUM_EPOCHS = 10
IMAGE_SIZE = 160

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# total train and validation images used for training
trainAug = ImageDataGenerator()
valAug = ImageDataGenerator()

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    TEST_PATH,
    class_mode="categorical",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)


# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
print("[INFO] preparing model...")
baseModel = ResNet50(weights="imagenet", include_top=False,
                     input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

print("ResNet model summary: ")
print(baseModel.summary())

# construct the head of the model that will be placed on top of the loaded network
# This will be the training part of the network

########################################################################


headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

print("Basemodel summary: ")
print(model.summary())


########################################################################


# loop over all layers in the base model and freeze them so they will
# BE updated during the training process

for layer in baseModel.layers:
    layer.trainable = True

# compile the model
opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', patience=7)
# train the model
print("[INFO] training model...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // batch_size,
    validation_data=valGen,
    validation_steps=totalVal // batch_size,
    epochs=NUM_EPOCHS)  # callbacks=[es])

# ##########################################################################
# ### This part of code for saving the model to disk
# ########################################################################

print("[INFO] saving model...")
# model.save(MODEL_NAME, save_format="h5")


# ##########################################################################
# ### This part of code plotting training loss and accuracy
# ########################################################################

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Training loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="validation loss")
plt.title("Training Loss for GASF")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt. grid(False)
plt.show()


plt.plot(np.arange(0, N), H.history["accuracy"], label="Training acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="Validation acc")
plt.title("Accuracy for GASF")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt. grid(False)
plt.show()
# plt.savefig(args["plot"])


# ##########################################################################
# ### This part of code for predicting testing images
# ########################################################################


# reset the testing generator and then use our trained model to
# make predictions on the datas
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
                                   steps=(totalTest // batch_size) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
                            target_names=testGen.class_indices.keys()))

# # Making the Confusion Matrix
cm = confusion_matrix(testGen.classes, predIdxs)
print('confusion matrix: ')
print(cm)


end = datetime.datetime.now()

print("Training completed after {} ".format(end-start))
