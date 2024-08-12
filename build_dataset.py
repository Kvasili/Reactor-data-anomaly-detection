'''

    This splits the data into training and validation dataset for the training procedure
    And a testing dataset used after the training to evaluate the model


    USAGE
    python build_dataset.py --path1 ./SCRAM_usecase_GASF_small --path2 ./SCRAM_training_GASF_small 

'''


from imutils import paths
import random
import shutil
import os
import argparse
import imutils


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p1", "--path1", required=True,
                help="path for the intial data")
ap.add_argument("-p2", "--path2", required=True,
                help="path for the training data")
args = vars(ap.parse_args())
print((args))

INPUT_DATASET = args['path1']

if (os.path.isdir(INPUT_DATASET)):
    print('ok !')
else:
    os.makedirs(args['path1'])


BASE_PATH = args['path2']
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])


# percentage of the total data used for training
TRAIN_SPLIT = 0.7

# Number of validation data will be a percentage of training data
# 20% of training data will be used for validation
VAL_SPLIT = 0.2

# path to all input images
# shuffle them
imagePaths = list(paths.list_images(INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)


# compute the training and testing split
i = int(len(imagePaths) * TRAIN_SPLIT)
trainPaths = imagePaths[:i]
print('number of training images {}'.format(int(len(trainPaths))))
testPaths = imagePaths[i:]
print('number of testing images {}'.format(int(len(testPaths))))

# we'll be using part of the training data for validation
i = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]


# define the datasets that we'll be building
datasets = [
    ("training", trainPaths, TRAIN_PATH),
    ("validation", valPaths, VAL_PATH),
    ("testing", testPaths, TEST_PATH)
]


# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    # show which data split we are creating
    print("[INFO] building '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)

        # loop over the input image paths
        for inputPath in imagePaths:
            # extract the filename of the input image along with its
            # corresponding class label
            filename = inputPath.split(os.path.sep)[-1]
            label = inputPath.split(os.path.sep)[-2]

            # build the path to the label directory
            labelPath = os.path.sep.join([baseOutput, label])

            # if the label output directory does not exist, create it
            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)

            # construct the path to the destination image and then copy
            # the image itself
            p = os.path.sep.join([labelPath, filename])
            shutil.copy2(inputPath, p)


print('Process completed !')
