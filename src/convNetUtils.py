import numpy as np
from rasterio.plot import reshape_as_image
from math import floor
import matplotlib.pyplot as pl
from sklearn.utils import class_weight
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils import np_utils, to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.layers.experimental.preprocessing import Rescaling
from keras.preprocessing.image import ImageDataGenerator


# Adapted from split_dataset of MLG course
def split_dataset(dataset, train_test_ratio=0.8):
    np.random.shuffle(dataset)
    nb_train = int(len(dataset) * train_test_ratio)
    return dataset[:nb_train], dataset[nb_train:]


def k_fold_indices(dataset, k=5):
    # subdivise dataset in k folds
    folds_indices = []
    fold_size = floor(len(dataset) / k)
    nb_larger_folds = len(dataset) % k
    fold_start = 0

    for fold_nb in range(k):
        current_fold_size = fold_size

        if fold_nb < nb_larger_folds:
            current_fold_size += 1

        fold_end = fold_start + current_fold_size
        folds_indices.append((fold_start, fold_end))
        fold_start = fold_end

    return folds_indices


def create_model(image_width, image_height, image_depth, nb_outputs):
    inputs = Input(shape=(image_width, image_height, image_depth))

    # Add layers of convolution and pooling
    model_layers = Rescaling(1./2**16)(inputs)
    model_layers = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu")(model_layers)
    model_layers = Conv2D(filters=8, kernel_size=(3, 3), padding="same", activation="relu")(model_layers)
    model_layers = MaxPooling2D(pool_size=(2, 2))(model_layers)
    model_layers = Conv2D(filters=16, kernel_size=(2, 2), padding="same", activation="relu")(model_layers)
    model_layers = Conv2D(filters=16, kernel_size=(2, 2), padding="same", activation="relu")(model_layers)
    model_layers = MaxPooling2D(pool_size=(2, 2))(model_layers)
    model_layers = Flatten(name='flat')(model_layers)
    model_layers = Dense(256, activation='relu')(model_layers)
    model_layers = Dropout(0.5)(model_layers)
    outputs = Dense(nb_outputs, activation='softmax')(model_layers)

    model = Model(inputs=inputs, outputs=outputs)

    # Specify optimizer and loss function
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_test, Y_test, class_weights, epochs, steps_per_epoch):
    # Define data generator arguments
    data_gen_args = dict(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # create data generator
    datagen = ImageDataGenerator(**data_gen_args)
    datagen.fit(X_train)
    train_datagen = datagen.flow(X_train, Y_train, batch_size=32)

    # Define fit arguments
    fit_args = dict(
        x=train_datagen,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
    )

    return model.fit(**fit_args)


def split_fold_into_train_test_sets(dataset, fold_start, fold_end, bands, labels_names, nb_classes):
    train = dataset[fold_start:fold_end]
    test = dataset[:fold_start] + dataset[fold_end:]

    X_train = images_from_dataset(train, bands)
    y_train = labels_from_dataset(train, labels_names)
    Y_train = to_categorical(y_train, num_classes=nb_classes)

    X_test = images_from_dataset(test, bands)
    y_test = labels_from_dataset(test, labels_names)
    Y_test = to_categorical(y_test, num_classes=nb_classes)

    return X_train, y_train, Y_train, X_test, y_test, Y_test


def cross_validation(dataset, bands, labels, labels_names, epochs, k=5):
    np.random.shuffle(dataset)

    histories = []

    # images have all the same shapes, take the shape of the first image
    image_width = len(dataset[0][1][0][0])
    image_height = len(dataset[0][1][0])
    image_depth = len(bands)

    mean_loss = 0
    mean_accuracy = 0

    for i, fold_indices in enumerate(k_fold_indices(dataset, k)):
        X_train, y_train, Y_train, X_test, y_test, Y_test = split_fold_into_train_test_sets(
            dataset, fold_indices[0], fold_indices[1], bands, labels_names, len(labels)
        )

        # Compute each classes weight
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(class_weights))

        model = create_model(image_width, image_height, image_depth, len(labels))

        # Print the model summary, only one time
        if i == 0:
            model.summary()

        print("\nFold", i, ":\n--------\n")

        # fit model
        history = train_model(model, X_train, Y_train, X_test, Y_test, class_weights, epochs, len(y_train) / 32)
        histories.append(history)

        # Evaluate model
        score = model.evaluate(X_test, Y_test, verbose=0)

        mean_loss += score[0] / k
        mean_accuracy += score[1] / k

    return mean_loss, mean_accuracy, histories


def images_from_dataset(dataset, bands):
    # min_per_band, max_per_band = compute_min_max_per_channel(dataset, bands)

    # Filter bands
    dataset = [
        [
            # normalize(img[1][band], min_per_band[i], max_per_band[i]) for i, band in enumerate(bands)
            img[1][band] for i, band in enumerate(bands)
        ] for img in dataset
    ]

    return np.array([reshape_as_image(img) for img in dataset])


def labels_from_dataset(dataset, labels):
    return np.array([labels.index(img[0]) for img in dataset])


# def compute_min_max_per_channel(dataset, bands):
#     min_per_channel = []
#     max_per_channel = []
#
#     for band in bands:
#         band_min = inf
#         band_max = -inf
#
#         for img in dataset:
#             img_values = img[1]
#
#             tmp_min = np.min(img_values[band])
#             tmp_max = np.max(img_values[band])
#
#             if tmp_min < band_min:
#                 band_min = tmp_min
#
#             if tmp_max > band_max:
#                 band_max = tmp_max
#
#         min_per_channel.append(band_min)
#         max_per_channel.append(band_max)
#
#     return min_per_channel, max_per_channel

#
# def normalize(values, min_val, max_val):
#     return (values - min_val) / (max_val - min_val)


# This function generates a colored confusion matrix.
# Adapted from plot_confusion_matrix of MLG course
def plot_confusion_matrix(confmatrix, labels_names, ax=None):
    if ax is None:
        ax = pl.subplot(111)

    ax.matshow(confmatrix, interpolation='nearest', cmap=pl.cm.Blues)

    for i in range(confmatrix.shape[0]):
        for j in range(confmatrix.shape[1]):
            ax.annotate(str(confmatrix[i, j]), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8)
    ax.set_xticks(np.arange(confmatrix.shape[0]))
    ax.set_xticklabels([labels_names[label] for label in range(confmatrix.shape[0])], rotation='vertical')
    ax.set_yticks(np.arange(confmatrix.shape[1]))
    _ = ax.set_yticklabels([labels_names[label] for label in range(confmatrix.shape[1])])
    ax.set_xlabel('predicted label')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('true label')
