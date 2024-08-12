
'''

    converting timeseries to image using the GAF method


    USAGE
    python GASF_to_images.py --path /home/radians/Documents/kostasV/series_to_images/scram_real_data.csv --folder Normal

    python GASF_to_images.py --path ../csv_data/scram_real_data_small.csv --folder Normal_test


    check:
    https://pyts.readthedocs.io/en/0.10.0/modules/image.html

'''

import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField as GAF
import pandas as pd
import os
import numpy as np
import datetime
import argparse

print('###################################    START    ##################################')

start = datetime.datetime.now()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path for the intial data")
ap.add_argument("-f", "--folder", required=True,
                help="folder to save the data")
args = vars(ap.parse_args())
print((args))


directory = args['path']

df = pd.read_csv(directory)
print(df.head())
print('************************')
print('INFO for the whole dataset\n')
print('Data dimensions: {}'.format(df.shape))
# print(df.describe().T)
print('***********************')


# ## Remove the first feature with no sensor data
df = df.iloc[:, 2:df.shape[1]]
print('Dimensions after removing first column: ')
print(df.shape)
print('***********************')


# ## Remove the first two features with no sensor data
# ## Remove 'Unnamed: 0' and 'index' feature for Normal data
# ## Remove 'Unnamed: 0' and 'Unnamed: 0.1' for Abnormal data

# ## Remove columns with no data
# index
# 0-15 channel error
# 16-31 channel error status
# error detection status 2

df = df.drop(['0-15 channel error', '16-31 channel error status',
             'error detection status 2'], axis=1)  # index'


print('Number of rows with at least one Null variable for Normalized data after null replacement')
print(df.isnull().any(axis=1).sum())
print('Total Null values for Normalized data: ')
print(df.isnull().sum().sum())


print('Dimensions after removing the columns: ')
print(df.shape)
print('***********************')


# Make dir to save the images
if not os.path.exists(args['folder']):
    os.makedirs(args['folder'])

WINDOW = 20
counter = 0
for index in range(0, len(df)-WINDOW):

    x_train = df.iloc[index:index+WINDOW]
    # print(x_train)

    # Data need to be in format (N x T), T: timeseries
    x_train_T = np.transpose(x_train.to_numpy())

    # # Gramian Angular Field (GAF) transform
    gasf = GAF(image_size=WINDOW, method='s')
    X_gasf = gasf.fit_transform(x_train_T)

    canvas = np.zeros((8*WINDOW, 8*WINDOW))
    # stack images in an 8x8 grid
    for i in range(8):
        for j in range(8):
            img = X_gasf[i*8+j]
            x_start = j*WINDOW
            x_end = (j+1)*WINDOW
            y_start = i*WINDOW
            y_end = (i+1)*WINDOW
            canvas[y_start:y_end, x_start:x_end] = img

    plt.imsave(args['folder']+'/' +
               '{}_GASF_{}.jpeg'.format(args['folder'], index), canvas)
    counter += 1


end = datetime.datetime.now()
duration = end-start

print('Process completed after {}'.format(duration))
print('Number of images created: {}'.format(counter))
