from rasterio.plot import reshape_as_image
from math import floor
import matplotlib.pyplot as pl
from sklearn.utils import class_weight
from sklearn import metrics as me
from keras.models import clone_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas
import geopandas as gpd
import spacv
import keras
import numpy as np


class Metrics(keras.callbacks.Callback):
    """
    Keras callback to provides additional metrics.
    It logs f1-score of the train and validation after each epoch. Those metrics can be used by the early stopping.
    Code inspired by: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    """

    def __init__(self, train, validation):
        """
        Initialize callback
        :param train: the train set as a tuple (train set values, correct labels)
        :param validation: the validation set as (test set values, correct labels)
        """

        super(Metrics, self).__init__()
        self.train = train
        self.validation = validation

    def on_epoch_end(self, epoch, logs={}):
        target_train = np.argmax(np.asarray(self.train[1]), axis=-1)
        predicted_train = np.argmax(np.asarray(self.model.predict(self.train[0])), axis=-1)
        f1_score_train = me.f1_score(target_train, predicted_train, average="macro")

        target_validation = np.argmax(np.asarray(self.validation[1]), axis=-1)
        predicted_validation = np.argmax(np.asarray(self.model.predict(self.validation[0])), axis=-1)
        f1_score_val = me.f1_score(target_validation, predicted_validation, average="macro")

        logs['f1_score_train'] = f1_score_train
        logs['f1_score_val'] = f1_score_val

        return


def hold_out_split_dataset(dataset, train_test_ratio=0.8):
    """
    Create an split of the dataset by holding out a ratio of the dataset
    to be used as test set.
    Adapted from split_dataset of MLG course
    :param dataset: the dataset to split
    :param train_test_ratio: the wanted ratio between train and test set
    :return: the train set and the test set
    """
    np.random.shuffle(dataset)
    nb_train = int(len(dataset) * train_test_ratio)
    return dataset[:nb_train], dataset[nb_train:]


def k_fold_indices(dataset, k=5):
    """
    Computes folds indices
    :param dataset: the dataset
    :param k: the number of fold
    :return: an array with the fold start and end indices
    """

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


def train_model(model, X_train, Y_train, X_test, Y_test, class_weights, epochs, steps_per_epoch, early_stopping=False):
    """
    Train a Keras neural network model
    :param model: the Keras neural network model
    :param X_train: the image train set
    :param Y_train: the correct labels of train set in one hot encoding
    :param X_test: the image test set
    :param Y_test: the correct labels of test set in one hot encoding
    :param class_weights: weight of each class. If classes are imbalanced it will gives more importance
    to underrepresented classes
    :param epochs: the number of epochs
    :param steps_per_epoch: the number of steps per epoch
    :param early_stopping: defines if early stopping is used
    :return: the train history
    """

    # Define data generator arguments
    data_gen_args = dict(
        rotation_range=45,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    batch_size = 32

    # create data generator
    datagen = ImageDataGenerator(**data_gen_args)
    datagen.fit(X_train)
    train_datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)

    callbacks = [Metrics(train=(X_train, Y_train), validation=(X_test, Y_test))]

    if early_stopping:
        callbacks.append(EarlyStopping(monitor='f1_score_val', patience=150, mode="max"))

    # Define fit arguments
    fit_args = dict(
        x=train_datagen,
        epochs=epochs,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_test, Y_test),
        callbacks=callbacks
    )

    return model.fit(**fit_args)


def split_fold_into_train_test_sets(dataset, fold_start, fold_end, bands, labels_names):
    """
    Get train and test sets from a given fold
    :param dataset: the dataset
    :param fold_start: start position of the fold in the dataset
    :param fold_end: end position of the fold in the dataset
    :param bands: the bands to use
    :param labels_names: the names of the labels to keep in the dataset
    :return:
    the image train set,
    the correct labels of test set,
    the correct labels of train set in one hot encoding,
    the image test set,
    the correct labels of test set,
    the correct labels of test set in one hot encoding
    """

    nb_classes = len(labels_names)

    train = dataset[fold_start:fold_end]
    test = dataset[:fold_start] + dataset[fold_end:]

    X_train = images_from_dataset(train, bands)
    y_train = labels_from_dataset(train, labels_names)
    Y_train = to_categorical(y_train, num_classes=nb_classes)

    X_test = images_from_dataset(test, bands)
    y_test = labels_from_dataset(test, labels_names)
    Y_test = to_categorical(y_test, num_classes=nb_classes)

    return X_train, y_train, Y_train, X_test, y_test, Y_test


def train_and_evaluate_fold(X_test, X_train, Y_test, Y_train, y_test, y_train, epochs, fold, model, nb_labels, early_stopping=False, new_model=True):
    """
    Train a fold and gives back results
    :param X_test: the image test set
    :param X_train: the image train set
    :param Y_test: the correct labels of test set in one hot encoding
    :param Y_train: the correct labels of train set in one hot encoding
    :param y_test: the correct labels of test set
    :param y_train: the correct labels of test set
    :param epochs: the number of epochs
    :param fold: the fold number
    :param model: the Keras neural network model
    :param nb_labels: the number of labels
    :param early_stopping: defines if there is early stopping
    :param new_model: defines if model should be reset or if the given model weights should be kept
    :return: history, confusion matrix, accuracy and loss
    """

    fold_size = len(y_train)

    # Compute each classes weight
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    if new_model:
        # clone given model without keeping the layers weights
        current_model = clone_model(model)

        # Specify optimizer and loss function
        current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        current_model = model

    print("\nFold ", (fold + 1), ":\n--------\n")

    # fit model
    history = train_model(
        current_model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        class_weights,
        epochs,
        fold_size / 32,
        early_stopping
    )

    # Evaluate model
    score = current_model.evaluate(X_test, Y_test, verbose=0)
    loss = score[0]
    accuracy = score[1]

    # Add current confusion matrix to the global confusion matrix
    pred = current_model.predict_on_batch(X_test)
    pred = np.argmax(pred, axis=-1)

    conf_matrix = me.confusion_matrix(y_test, pred, labels=np.arange(nb_labels))

    return history, conf_matrix, accuracy, loss


def cross_validation(model, dataset, bands, labels, epochs, nb_cross_validations=1, k=5, early_stopping=False):
    """
    This is a function to do cross validation on a given keras model.
    :param model: the Keras neural network model
    :param dataset: the dataset, typically created with make_dataset_from_raster_files
    :param bands: an array of the position of the bands to use. ex: [3, 2, 1] will select bands Red, Green, Blue
    if the dataset contains images with all bands. Bands positions start at zero.
    :param labels: an array of selected labels. Those labels should be entries of Label enum defined in labelsUtils.py
    :param epochs: the number of epochs
    :param nb_cross_validations: the number of time to repeat the cross validation.
    :param k: the number of folds
    :param early_stopping: defines if early stopping is used
    :return: mean loss, mean accuracy, array of each history and the confusion matrix
    """
    labels_names = [label.name for label in labels]
    nb_labels = len(labels_names)
    histories = []
    mean_loss = 0
    mean_accuracy = 0
    total_conf_matrix = np.zeros((len(labels_names), len(labels_names)))

    model.summary()

    for validation in range(nb_cross_validations):
        np.random.shuffle(dataset)

        for fold, fold_indices in enumerate(k_fold_indices(dataset, k)):
            X_train, y_train, Y_train, X_test, y_test, Y_test = split_fold_into_train_test_sets(
                dataset, fold_indices[0], fold_indices[1], bands, labels_names
            )

            history, conf_matrix, accuracy, loss = train_and_evaluate_fold(
                X_test, X_train, Y_test, Y_train, y_test, y_train,
                epochs, fold, model, nb_labels,
                early_stopping
            )

            fold_size = len(y_train)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_loss, mean_accuracy, histories, total_conf_matrix


def cross_validation_with_metrics_evolution(model, dataset, bands, labels, epochs, epochs_per_metrics, nb_cross_validations=1, k=5):
    """
    Cross validation that stops each n epochs to computes some metrics.
    Total number of epochs are divided by a number of epochs per metrics to get a number of metrics.
    Each metrics are shared between folds.
    :param model: Keras neural network model
    :param dataset: dataset, typically created with make_dataset_from_raster_files
    :param bands: an array of the position of the bands to use. ex: [3, 2, 1] will select bands Red, Green, Blue
    if the dataset contains images with all bands. Bands positions start at zero.
    :param labels: an array of selected labels. Those labels should be entries of Label enum defined in labelsUtils.py
    :param epochs: number of epochs
    :param epochs_per_metrics: number of epochs per metrics
    :param nb_cross_validations: number of time to repeat the cross validation.
    :param k: the number of folds
    :return: Array of mean losses, mean accuracies, histories and confusion matrices. Each position in this array
    """

    assert epochs % epochs_per_metrics == 0, "epochs should be dividable by epochs_per_metrics"
    assert epochs_per_metrics < epochs, "epochs per metrics should be inferior to epochs"
    assert epochs_per_metrics > 0, "epochs per metrics should be superior to 0"

    labels_names = [label.name for label in labels]
    nb_labels = len(labels_names)
    histories = []
    # Each metrics are shared between folds
    nb_metrics = epochs // epochs_per_metrics
    mean_accuracies = np.zeros(nb_metrics)
    mean_losses = np.zeros(nb_metrics)
    total_conf_matrices = np.zeros((nb_metrics, len(labels_names), len(labels_names)))

    model.summary()

    for validation in range(nb_cross_validations):
        np.random.shuffle(dataset)

        for fold, fold_indices in enumerate(k_fold_indices(dataset, k)):
            X_train, y_train, Y_train, X_test, y_test, Y_test = split_fold_into_train_test_sets(
                dataset, fold_indices[0], fold_indices[1], bands, labels_names
            )

            # We must reinitialize the model each fold!
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            for metric in range(nb_metrics):
                history, conf_matrix, accuracy, loss = train_and_evaluate_fold(
                    X_test, X_train, Y_test, Y_train, y_test, y_train,
                    epochs_per_metrics, fold, current_model, nb_labels, early_stopping=False, new_model=False
                )

                fold_size = len(y_train)

                histories.append(history)
                total_conf_matrices[metric] += conf_matrix
                mean_accuracies[metric] += accuracy * fold_size / (len(dataset) * nb_cross_validations)
                mean_losses[metric] += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_losses, mean_accuracies, histories, total_conf_matrices


def spatial_cross_validation(model, dataset, bands, labels, epochs, nb_cross_validations=1, early_stopping=False):
    """
    Spatial cross validation to train on a region and validate with another
    :param model:  Keras neural network model
    :param dataset: dataset, typically created with make_dataset_from_raster_files
    :param bands: an array of the position of the bands to use. ex: [3, 2, 1] will select bands Red, Green, Blue
    if the dataset contains images with all bands. Bands positions start at zero.
    :param labels: an array of selected labels. Those labels should be entries of Label enum defined in labelsUtils.py
    :param epochs: number of epochs
    :param nb_cross_validations: number of time to repeat the cross validation
    :param early_stopping: defines if early stopping is used
    :return: mean loss, mean accuracy, array of each history and the confusion matrix
    """

    labels_names = [label.name for label in labels]

    nb_labels = len(labels_names)
    histories = []
    mean_loss = 0
    mean_accuracy = 0
    total_conf_matrix = np.zeros((len(labels_names), len(labels_names)))

    model.summary()

    df = pandas.DataFrame(dataset,  columns=['labels_names', 'images', 'coords'])
    labels_coordinates = gpd.GeoDataFrame(df, geometry='coords') # spacv requests a dataframe

    for validation in range(nb_cross_validations):
        fold = 0

        skcv = spacv.SKCV(n_splits=5, buffer_radius=0.1).split(labels_coordinates)

        for train, test in skcv:
            train = [img for i, img in enumerate(dataset) if i in train]
            test = [img for i, img in enumerate(dataset) if i in test]

            X_train = images_from_dataset(train, bands)
            y_train = labels_from_dataset(train, labels_names)
            Y_train = to_categorical(y_train, num_classes=nb_labels)

            X_test = images_from_dataset(test, bands)
            y_test = labels_from_dataset(test, labels_names)
            Y_test = to_categorical(y_test, num_classes=nb_labels)

            history, conf_matrix, accuracy, loss = train_and_evaluate_fold(
                X_test, X_train, Y_test, Y_train, y_test, y_train,
                epochs, fold, model, nb_labels,
                early_stopping
            )

            fold_size = len(y_train)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

            fold += 1

    return mean_loss, mean_accuracy, histories, total_conf_matrix


def images_from_dataset(dataset, bands):
    """
    Extract images from a dateset into a numpy array, while taking only certain bands.
    Images format is converted from (bands, rows, columns) to (rows, columns, bands)
    :param dataset: the dataset
    :param bands: the bands to keep
    :return: an images array
    """
    # Filter bands
    dataset = [
        [
            img[1][band] for i, band in enumerate(bands)
        ] for img in dataset
    ]

    return np.array([reshape_as_image(img) for img in dataset])


def labels_from_dataset(dataset, labels):
    """
    Extract labels from a dataset into a numpy array
    :param dataset: the dataset
    :param labels: the labels to keep
    :return: a labels array
    """
    return np.array([labels.index(img[0]) for img in dataset])


def plot_confusion_matrix(confmatrix, labels_names, ax=None):
    """
    This function generates a colored confusion matrix.
    Code has been taken and adapted from plot_confusion_matrix of MLG course
    :param confmatrix: the confusion matrix
    :param labels_names: labels names
    :param ax: matplotlib axes object to plot to
    """
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


# def add_ndvi_to_dataset(dataset):
#     for i, img in enumerate(dataset):
#         red = np.asarray(img[1][3])
#         nir = np.asarray(img[1][4])
#         ndvi = (nir - red) / (nir + red)
#         dataset[i][1].append(ndvi.tolist())
#
#     return dataset
#
#
# def add_mndwi_to_dataset(dataset):
#     for i, img in enumerate(dataset):
#         green = np.asarray(img[1][2])
#         swir = np.asarray(img[1][5])
#         mndwi = (green - swir) / (green + swir)
#         dataset[i][1].append(mndwi.tolist())
#
#     return dataset
