from keras.models import Model
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, merge, BatchNormalization, Input
from keras.layers.core import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
import tensorflow as tf
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import argparse

def driveLogGenerator(log_df, batch_size=10, start_idx=0, max_batch=10000, correction=0.25, with_flip=True):
    batch_counter = 0
    while batch_counter < max_batch: 
        start_idx = start_idx if start_idx < len(log_df) else start_idx - len(log_df)
        end_idx = min(len(log_df), start_idx + batch_size - 1)
        batch_images = []
        steers = []
        steer_col = [c for c in log_df if 'steer' in c][0]
        if with_flip: 
            for idx, row in log_df.ix[start_idx:end_idx].iterrows():
                im_center = load_and_cvtColor(row['center'])
                im_left = load_and_cvtColor(row['left'])
                im_right = load_and_cvtColor(row['right'])
                batch_images.extend([
                    im_center, im_left, im_right, 
                    np.fliplr(im_center), np.fliplr(im_left), np.fliplr(im_right)
                ])
                steers.extend([
                    row[steer_col], row[steer_col]+correction, row[steer_col]-correction, 
                    row[steer_col], row[steer_col]-correction, row[steer_col]+correction              
                ])

            start_idx += batch_size
            batch_counter += 1
            yield np.array(batch_images), steers
        else: 
            for idx, row in log_df.ix[start_idx:end_idx].iterrows():
                im_center = load_and_cvtColor(row['center'])
                im_left = load_and_cvtColor(row['left'])
                im_right = load_and_cvtColor(row['right'])
                batch_images.extend([
                    im_center, im_left, im_right
                ])
                steers.extend([
                    row[steer_col], row[steer_col]+correction, row[steer_col]-correction           
                ])

            start_idx += batch_size
            batch_counter += 1
            yield np.array(batch_images), steers

def Conv2D_BN(input_x, filters, rows, cols, border_mode='same', strides=(1, 1)):
    """ Combine Convolution2D and BatchNormalization
    """
    input_x = Convolution2D(filters, rows, cols,
                            subsample=strides,
                            activation='elu',
                            border_mode=border_mode)(input_x)
    input_x = BatchNormalization()(input_x)
    return input_x 


def model(ch=3, row=160, col=320):
    img_input = Input(shape=(row, col, ch))
    x = Lambda(lambda x: x / 255.0 - 0.5)(img_input)
    x = Cropping2D(cropping=((70, 25), (0, 0)))(x)
    x = Conv2D_BN(x, 24, 5, 5, strides=(2, 2))
    x = Conv2D_BN(x, 36, 5, 5, strides=(2, 2))
    x = Conv2D_BN(x, 64, 5, 5, strides=(2, 2))
    x = Conv2D_BN(x, 256, 3, 3)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = Dense(1)(x)
    return Model(img_input, x)

if __name__ == __main__:
    # Parse argument
    ps = argparse.ArgumentParser()
    ps.add_argument('--batch_size', default=128, type=int, help='training batch size')
    ps.add_argument('--epoches', default=100, type=int, help='number of training epoches')
    ps.add_argument('--samples_per_epoch', default=512, type=int, help='number of samples per epoch')
    ps.add_argument('--model_file', default='model/model.h5', type=str, help='name of the output model file')
    args = ps.parse_args()

    model = model()
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    log_files = [
        '../lap2/driving_log.csv',
        '../track2_lap1/driving_log.csv'
        '../provided_data/driving_log.csv'
    ]

    log_cols = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed']
    for log_file in log_files: 
        log_df = pd.read_csv(log_file, header=None)
        log_df.columns = log_cols
        print("Training with generated data : {}".format(log_file))
        print("-"*78)
        model.fit_generator(
            driveLogGenerator(log_df, batch_size=int(args.batch_size)), 
            samples_per_epoch = int(args.samples_per_epoch), nb_epoch = int(args.epoches), verbose=1, 
            callbacks=[], validation_data=driveLogGenerator(log_df), nb_val_samples=100, nb_worker=1
        )
        print("Saving model to disk at : {}".format(model_file))
        print("-"*78)
        model.save(os.path.join(model_output_path, 'nv_model.h5'))

    print("Training completed !")
