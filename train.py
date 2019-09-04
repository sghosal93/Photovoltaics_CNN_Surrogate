"""
Trains the CNN surrogate model from scratch (no pretraining)
run it as : python model.py

Required packages: keras (on top of tensorflow), developed with Python 2.7

Replace "****your desired location****" with appropriate directory location information

"""

import h5py
import numpy as np
import keras

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import json
import pickle
import gc
import scipy.io
import time
import os

size_morphology = np.array([101, 101]) # Input dimensions
n_labels = 10 # Number of classes (bins) for classification

################################
# Function for saving a figure #
################################

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")

########################################################################################

def load_data_labels(filename):
    with h5py.File(filename, 'r') as h5f:
        h5_data = np.asarray(list(h5f['morphology']))
        h5_labels = np.asarray(list(h5f['labels']))
    print('data and labels are loaded!')
    return h5_data, h5_labels


def flip_morphology(phi, size=size_morphology):
    r"""
    Performs a left-right flip of the morphology, of the
    size size_morphology
    Example:
    /  1 2 3   \            /  3 2 1  \
    |  4 5 6   |            |  6 5 4  |
    |  7 8 9   |      ->    |  9 8 7  |
    \  0 0 2   /            \  2 0 0  /
    :param phi: numpy array of size : np.prod(size_morphology)
    :param size: size of morphology, defaulted to size_morphology
    :return: numpy array of the same size as phi, but flipped
    """
    phi_reshape = np.reshape(phi, size)
    phi_flip = np.reshape(np.fliplr(phi_reshape), (1, np.prod(size)))
    return phi_flip


def flip_data(data):
    data_flipped = data
    for i in np.arange(data.shape[0]):
        data_flipped[i] = flip_morphology(data[i], size=size_morphology)
    return data_flipped


def create_cnn_model():
    col, row = size_morphology      # size of each image
    ch = 1                          # number of channels
    model = Sequential()
   
    # CNN Surrogate Model

    model.add(Conv2D(16, (5, 5), strides = (1, 1),  activation='relu', input_shape=(col, row, ch)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (5, 5), strides = (1, 1),  activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), strides = (1, 1),  activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), strides = (1, 1),  activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(.5)) 

    model.add(Dense(n_labels, activation='softmax'))
    model.summary()

    opt = Adam(lr=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  #metrics=['accuracy', 'fmeasure', 'precision', 'recall'])
                  metrics=['accuracy'])

    #plot(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # Save model to json file
    json_string = model.to_json()
    with open('****your desired location****/model.json', 'w') as f:
        json.dump(json_string, f)

    return model


def bin_labels(labels_to_bin, num_labels=n_labels):
    
    # performs a uniform division of labels among all the labels
    binned_labels = np.zeros((np.asarray([labels_to_bin]).transpose().shape[0], num_labels))
    bins = np.linspace(start=np.amin(labels_to_bin), stop=np.amax(labels_to_bin), num=num_labels+1)
    for label_num in np.arange(labels_to_bin.shape[0]):  # iterate through all the rows
        binned_labels[label_num, np.searchsorted(bins, labels_to_bin[label_num])-1] = 1
    
    return np.asarray(binned_labels)


if __name__ == '__main__':
    if_restart = False
    # if_restart = True

    # load data from h5 file
    raw_data, raw_labels = load_data_labels(
        'morphology_data_global.h5')
    # flip data: the labels for the flipped data will be the same from raw_data
    flipped_data = flip_data(raw_data)

    # extract the label we want : Jsc is in 3rd column : look at load_and_save_data.py
    # for more information
    labels = raw_labels[:, 2]
    flip_labels = labels

    # find the outlier data: flags are located indices 6,7,8
    flags = raw_labels[:, 6:9]

    size_all_data = 0
    curr_row_num = 0
    indx = []
    for row in flags:
        if row[0] < 0.1 and row[1] < 0.01 and row[2] < 0.2:
            size_all_data += 1
            indx.append(curr_row_num)
        curr_row_num += 1

    indx = np.asarray(indx)
    # to consider flipped data
    size_all_data *= 2

    # all useful data:
    all_data = np.empty((size_all_data, np.prod(size_morphology)))
    all_data[np.arange(indx.shape[0])] = raw_data[indx]
    all_data[indx.shape[0] + np.arange(indx.shape[0])] = flipped_data[indx]

    all_labels = np.empty(size_all_data)
    all_labels[np.arange(indx.shape[0])] = labels[indx]
    all_labels[indx.shape[0] + np.arange(indx.shape[0])] = labels[indx]

    # categorized labels
    all_category_labels = bin_labels(all_labels)
    print('Labelling and Binning Done.')
    # reshape the data into images
    all_data_reshaped = np.reshape(all_data, (size_all_data, size_morphology[0], size_morphology[1], 1))
    # threshold into a binary image :
    all_data_reshaped = np.float16(all_data_reshaped > 0.5)

    x_train, x_test, y_train, y_test = train_test_split(all_data_reshaped, all_category_labels, test_size=0.33,
                                                        random_state=108)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.50, random_state=192)

    print('Printing Train, Validation and Test Data shape..................................')
    print('Training data:', x_train.shape)
    print('Validation data:', x_valid.shape)
    print('Test data:', x_test.shape)
    print('DONE printing Train, Validation and Test Data shape..................................')

    test_set_x = np.asarray(x_test)
    test_set_y = np.asarray(y_test)
    scipy.io.savemat('****your desired location****/test_set_x.mat', mdict={'test_set_x': test_set_x})
    scipy.io.savemat('****your desired location****/test_set_y.mat', mdict={'test_set_y': test_set_y})
    #exit() # <----------------------------------------------------------------------------------------------------------- CHECKPOINT
    if not if_restart:
        model = create_cnn_model()
        model.save('****your desired location****/model_architecture_thresholded.h5')
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=1),
            keras.callbacks.ModelCheckpoint(filepath='****your desired location****/threshold_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                            monitor='val_acc', save_best_only=True, verbose=1),
        ]
        train_history = model.fit(x_train, y_train, batch_size=128, epochs=100, verbose=1, validation_split=0,
                                  validation_data=(x_valid, y_valid), shuffle=True, callbacks=callbacks,
                                  class_weight=np.ones(n_labels))
        pickle.dump(train_history.history, open('****your desired location****/train_history_thresholded.p', 'wb'))
        model.save_weights("****your desired location****/model/model_thresholded.h5")
    else:   # load model architecture and weights
        # the backup files in this are the same as the files that were loaded.
        
        model = load_model('model_architecture_backup.h5')
        model.load_weights('weights_model_backup.h5')
        fileId = open('train_history_backup.p', 'rb')
        train_history = pickle.load(fileId)
        print('model architecture, final weights and  training history loaded !')
    gc.collect()

    y_test_predictions = model.predict(x=x_test)
    y_test_class = np.argmax(y_test, axis=1)
    y_pred_class = np.argmax(y_test_predictions, axis=1)
    
    preds = np.vstack(np.asarray(y_pred_class))
    scipy.io.savemat('****your desired location****/predictions.mat', mdict={'pred_y': preds})

    # confusion matrix:
    conf_matr = confusion_matrix(y_test_class, y_pred_class, labels=np.arange(n_labels))
    # plot confusion matrix:
    plt.imshow(conf_matr, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(n_labels)
    plt.xticks(tick_marks, np.arange(n_labels), rotation=45)
    plt.yticks(tick_marks, np.arange(n_labels))
    print(conf_matr)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('****your desired location****/conf_matrix_thresholded.png')
    plt.clf()
    
    # normalization
    cm_norm = conf_matr.astype('float') / conf_matr.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix:")
    print(cm_norm)
    print("Actual confusion matrix:")
    print(conf_matr)

    # save some results: accuracy, loss

    acc = train_history.history['acc']
    val_acc = train_history.history['val_acc']

    loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val_loss'])
    save("plots/losses", ext="png", close=False, verbose=True)
    plt.close()

    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc', 'val_acc'], loc=4)
    save("plots/accuracy", ext="png", close=False, verbose=True)
    plt.close()

