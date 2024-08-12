'''
     @Description 
     This code for altering the manual SCRAMS from 10s to every 20 seconds

    USAGE

    python extract_SCRAMS_fake.py --path ../csv_data/scrams_fake_data.csv --folder Abnormal2

'''

import matplotlib.pyplot as plt
import os
import pandas as pd
from pyts.image import GramianAngularField as GAF
import datetime
import numpy as np
import os
import argparse


start = datetime.datetime.now()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path for the data")
ap.add_argument("-f", "--folder", required=True,
                help="folder to save the images")
args = vars(ap.parse_args())
print((args))


if not os.path.exists(args['folder']):
    os.makedirs(args['folder'])


df = pd.read_csv(args['path'])
print('************************')
print('INFO for the whole dataset\n')
print('Data dimensions: {}'.format(df.shape))
# print(df.describe().T)
print('***********************')

print('INFO for the null cases:')
print('Total number of Null use cases: {}'.format(df.isnull().sum().sum()))

# ## Remove the first feature with no sensor data
df = df.iloc[:, 2:df.shape[1]]

# ## Remove columns with no data
# 0-15 channel error
# 16-31 channel error status
# error detection status 2IMAGE_SIZE

df = df.drop(['0-15 channel error', '16-31 channel error status',
             'error detection status 2'], axis=1)  # index'

# df.describe().T.to_csv('{}_data_statistics.csv'.format(args['folder']))#
print('Dimensions after removing the No data columns')
print(df.shape)
print('***********************')

# select all rows with NaN under the entire DataFrame
print('Number of rows with at least one Null variable: ')
print(df.isnull().any(axis=1).sum())
print('\n')
print('Null values across the rows:')
print(df.isnull().sum(axis=0))  # measures null values across the rows

print('\n')
print('Drop Null values')
df.dropna(axis=0, inplace=True)
print('Number of rows with at least one Null variable: ',
      df.isnull().any(axis=1).sum())
print('\n')


# counter for measuring Scram images of interest
counter = 0
for i in range(0, len(df)):  # df
    '''
        This loop for altering the MANUAL-SCRAM = 1.0 every 20 seconds instead of 10 seconds

    '''
    if (df.loc[i, 'manual-scram'] == 1.0):
        counter += 1
        if (counter % 2 == 0):

            df.loc[i, 'manual-scram'] = 0.0

print(df[df['manual-scram'] == 1.0])

WINDOW = 20
counter = 0
for index in range(0, len(df) - WINDOW):

    x_train = df.iloc[index:index+20]
    # print(x_train)

    # time series need to be in format N x T (T is the timestamp)
    x_train_T = np.transpose(x_train.to_numpy())

    # # Gramian Angular Field (GAF) transform
    gasf = GAF(image_size=20, method='s')
    X_gasf = gasf.fit_transform(x_train_T)

    canvas = np.zeros((8*20, 8*20))
    # stack images in an 8x8 grid
    for i in range(8):
        for j in range(8):
            img = X_gasf[i*8+j]  # 0-64
            x_start = j*20
            x_end = (j+1)*20
            y_start = i*20
            y_end = (i+1)*20
            canvas[y_start:y_end, x_start:x_end] = img

    plt.imsave(args['folder']+'/' +
               '{}_GASF_{}.jpeg'.format(args['folder'], index), canvas)
    counter += 1


end = datetime.datetime.now()
duration = end-start

print('Process completed after {}'.format(duration))
print('Number of images created: {}'.format(counter))
