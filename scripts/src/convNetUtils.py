from rasterio.plot import reshape_as_image
from math import floor
import matplotlib.pyplot as pl
from sklearn.utils import class_weight
from sklearn import metrics as me
from keras.models import clone_model
from keras.utils import to_categorical, Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import numpy as np
import pandas
import spacv

from albumentations import (
    Compose,
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    RandomRotate90,
)

MODEL_PATH = '../models/'

AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0,
        rotate_limit=0,
        p=0.8
    ),
    RandomRotate90(p=0.5),
])


class LandsatSequence(Sequence):
    """
    Code taken from https://medium.com/the-artificial-impostor/custom-image-augmentation-with-keras-70595b01aeac
    """

    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)


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
        :param validation: the validation set as (validation set values, correct labels)
        """

        super(Metrics, self).__init__()
        self._supports_tf_logs = True
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


def train_model(model, X_train, Y_train, X_validation, Y_validation, class_weights, epochs, steps_per_epoch,
                early_stopping=False, model_checkpoint_cb=None):
    """
    Train a Keras neural network model
    :param model: the Keras neural network model
    :param X_train: the image train set
    :param Y_train: the correct labels of train set in one hot encoding
    :param X_validation: the image validation set
    :param Y_validation: the correct labels of validation set in one hot encoding
    :param class_weights: weight of each class. If classes are imbalanced it will gives more importance
    to underrepresented classes
    :param epochs: the number of epochs
    :param steps_per_epoch: the number of steps per epoch
    :param early_stopping: defines if early stopping is used
    :return: the train history
    """
    batch_size = 32

    # create data generator
    # datagen = ImageDataGenerator(**data_gen_args)
    # datagen.fit(X_train)
    # train_datagen = datagen.flow(X_train, Y_train, batch_size=batch_size)
    train_datagen = LandsatSequence(X_train, Y_train, batch_size, AUGMENTATIONS)

    callbacks = [Metrics(train=(X_train, Y_train), validation=(X_validation, Y_validation))]

    if early_stopping:
        callbacks.append(EarlyStopping(monitor='f1_score_val', patience=100, mode="max"))

    if model_checkpoint_cb:
        callbacks.append(model_checkpoint_cb)

    # Define fit arguments
    fit_args = dict(
        x=train_datagen,
        epochs=epochs,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_validation, Y_validation),
        callbacks=callbacks
    )

    return model.fit(**fit_args)


def separate_data_into_images_and_labels(data, bands, labels_names):
    images = images_from_dataset(data, bands)
    labels = labels_from_dataset(data, labels_names)
    one_hot_labels = to_categorical(labels, num_classes=len(labels_names))

    return images, labels, one_hot_labels


def split_fold_into_train_validation_sets(dataset, fold_start, fold_end, bands, labels_names):
    """
    Get train and validation sets from a given fold
    :param dataset: the dataset
    :param fold_start: start position of the fold in the dataset
    :param fold_end: end position of the fold in the dataset
    :param bands: the bands to use
    :param labels_names: the names of the labels to keep in the dataset
    :return:
    the image train set,
    the correct labels of validation set,
    the correct labels of train set in one hot encoding,
    the image validation set,
    the correct labels of validation set,
    the correct labels of validation set in one hot encoding
    """

    train = dataset[fold_start:fold_end]
    validation = dataset[:fold_start] + dataset[fold_end:]

    X_train, y_train, Y_train = separate_data_into_images_and_labels(train, bands, labels_names)
    X_validation, y_validation, Y_validation = separate_data_into_images_and_labels(validation, bands, labels_names)

    return X_train, y_train, Y_train, X_validation, y_validation, Y_validation


def evaluate_model(model, X_test, Y_test, y_test, nb_labels):
    # Evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)
    loss = score[0]
    accuracy = score[1]

    # Predict labels on batch
    pred = model.predict_on_batch(X_test)
    pred = np.argmax(pred, axis=-1)

    # Confusion matrix
    conf_matrix = me.confusion_matrix(y_test, pred, labels=np.arange(nb_labels))

    return conf_matrix, accuracy, loss


def train_fold(X_validation, X_train, Y_validation, Y_train, y_validation, y_train, epochs, fold, validation, model,
               new_model=True, early_stopping=False, model_checkpoint_cb=None):
    """
    Train a fold
    :param X_validation: the image validation set
    :param X_train: the image train set
    :param Y_validation: the correct labels of validation set in one hot encoding
    :param Y_train: the correct labels of train set in one hot encoding
    :param y_validation: the correct labels of validation set
    :param y_train: the correct labels of validation set
    :param epochs: the number of epochs
    :param fold: the fold number
    :param validation: the validation number
    :param model: the Keras neural network model
    :param new_model: defines if model should be reset or if the given model weights should be kept
    :param early_stopping: defines if there is early stopping
    :param model_checkpoint_cb: a model checkpoint callback
    :return: history, trained model
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

    print("\nValidation ", (validation + 1), ", fold ", (fold + 1), ":\n---------------------------\n")

    # fit model
    history = train_model(
        current_model,
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        class_weights,
        epochs,
        fold_size / 32,
        early_stopping,
        model_checkpoint_cb,
    )

    return history, current_model


def cross_validation(model, dataset, bands, labels, epochs, nb_cross_validations=1, k=5, early_stopping=False,
                     with_model_checkpoint=False, model_name="model"):
    """
    :param model: the Keras neural network model
    :param dataset: the dataset, typically created with make_dataset_from_raster_files
    :param bands: an array of the position of the bands to use. ex: [3, 2, 1] will select bands Red, Green, Blue
    if the dataset contains images with all bands. Bands positions start at zero.
    :param labels: an array of selected labels. Those labels should be entries of Label enum defined in labelsUtils.py
    :param epochs: the number of epochs
    :param nb_cross_validations: the number of time to repeat the cross validation.
    :param k: the number of folds
    :param early_stopping: defines if early stopping is used
    :param model_name: name of the model, used to name the model file 
    :param with_model_checkpoint: defines if model checkpoint should be used.
    :return: mean loss, mean accuracy, array of each history and the confusion matrix
    """

    labels_names = [label.name for label in labels]
    nb_labels = len(labels_names)
    histories = []
    mean_loss = 0
    mean_accuracy = 0
    total_conf_matrix = np.zeros((len(labels_names), len(labels_names)))

    model.summary()

    model_checkpoint_cb = None

    if with_model_checkpoint:
        # Model checkpoint callback should be created here to avoid resetting the 'best' property of the model checkpoint.
        # By doing so, we ensure that model checkpoint is the best across all cross validations.
        model_file = MODEL_PATH + model_name + ".hdf5"
        model_checkpoint_cb = ModelCheckpoint(model_file, monitor='f1_score_val', verbose=0, save_best_only=True,
                                              mode='max')

    for validation in range(nb_cross_validations):
        np.random.shuffle(dataset)

        for fold, fold_indices in enumerate(k_fold_indices(dataset, k)):
            X_train, y_train, Y_train, X_validation, y_validation, Y_validation = split_fold_into_train_validation_sets(
                dataset, fold_indices[0], fold_indices[1], bands, labels_names
            )

            history, trained_model = train_fold(
                X_validation=X_validation,
                X_train=X_train,
                Y_validation=Y_validation,
                Y_train=Y_train,
                y_validation=y_validation,
                y_train=y_train,
                epochs=epochs,
                fold=fold,
                validation=validation,
                model=model,
                early_stopping=early_stopping,
                model_checkpoint_cb=model_checkpoint_cb,
                new_model=True,
            )

            conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation,
                                                         nb_labels)

            fold_size = len(y_train)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_loss, mean_accuracy, histories, total_conf_matrix


def cross_validation_with_metrics_evolution(model, dataset, bands, labels, epochs, epochs_per_metrics,
                                            nb_cross_validations=1, k=5):
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
            X_train, y_train, Y_train, X_validation, y_validation, Y_validation = split_fold_into_train_validation_sets(
                dataset, fold_indices[0], fold_indices[1], bands, labels_names
            )

            # We must reinitialize the model each fold!
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            for metric in range(nb_metrics):
                history, current_model = train_fold(
                    X_validation=X_validation,
                    X_train=X_train,
                    Y_validation=Y_validation,
                    Y_train=Y_train,
                    y_validation=y_validation,
                    y_train=y_train,
                    epochs=epochs,
                    fold=fold,
                    model=model,
                    early_stopping=False,
                    model_checkpoint_cb=None,
                    new_model=False,
                )

                conf_matrix, accuracy, loss = evaluate_model(current_model, X_validation, Y_validation, y_validation,
                                                             nb_labels)

                fold_size = len(y_train)

                histories.append(history)
                total_conf_matrices[metric] += conf_matrix
                mean_accuracies[metric] += accuracy * fold_size / (len(dataset) * nb_cross_validations)
                mean_losses[metric] += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_losses, mean_accuracies, histories, total_conf_matrices


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


def add_ndvi_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        ndvi = (nir - red) / (nir + red)
        ndvi = (ndvi + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(ndvi.tolist())


def add_bu_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        swir = np.asarray(img[1][5])
        ndbi = (swir - nir) / (swir + nir)
        ndvi = (nir - red) / (nir + red)
        bu = ndbi - ndvi
        bu = (bu + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(bu.tolist())


def add_mndwi_to_dataset(dataset):
    for i, img in enumerate(dataset):
        green = np.asarray(img[1][2])
        swir = np.asarray(img[1][5])
        mndwi = (green - swir) / (green + swir)
        mndwi = (mndwi + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(mndwi.tolist())


def add_evi2_to_dataset(dataset):
    for i, img in enumerate(dataset):
        red = np.asarray(img[1][3])
        nir = np.asarray(img[1][4])
        evi2 = 2.4 * (nir - red) / (nir + red + 1)
        evi2 = (evi2 + 1) / 2  # convert from [-1;1] to [0;1]
        dataset[i][1].append(evi2.tolist())


def spatial_separation_dataset(geo_df, labels):
    labels_names = [label.name for label in labels]

    nb_fold = 10
    fold_list = [[[] for _ in range(len(labels_names))] for _ in range(nb_fold)]
    labels_data = []

    # Make 10 folds with a train and validation set of each label
    for label_index, label_name in enumerate(labels_names):
        label_data = geo_df[geo_df['labels_names'] == label_name]
        labels_data.append(label_data)

        fold = 0

        for _, validation in spacv.SKCV(n_splits=nb_fold, buffer_radius=0.1).split(label_data):
            fold_list[fold][label_index] = validation

            fold += 1

    np.random.shuffle(fold_list)

    # We know that each validation set is unique
    # we choose two validation sets for the test set
    # and we keep other for the train and validation
    nb_grouped_fold = 2

    for i in range(0, nb_fold, nb_grouped_fold):
        current_selected_fold = [i, i+1]
        current_other = np.concatenate([np.arange(0, i), np.arange(i+2, nb_fold)])

        selected_fold = pandas.DataFrame()
        other = pandas.DataFrame()

        for label in range(len(labels_names)):
            selected_fold = selected_fold.append(
                labels_data[label].iloc[np.concatenate([list(fold_list[fold][label]) for fold in current_selected_fold]).ravel().tolist()]
            )

            other = other.append(
                labels_data[label].iloc[np.concatenate([list(fold_list[fold][label]) for fold in current_other]).ravel().tolist()]
            )

        yield selected_fold, other


def spatial_cross_validation(model, dataset, bands, labels, epochs, nb_cross_validations=1, early_stopping=False,
                             with_model_checkpoint=False, model_name="model"):
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
    :param model_name: Keras neural network model
    :param with_model_checkpoint: defines if model checkpoint should be used.
    :return: mean loss, mean accuracy, array of each history and the confusion matrix
    """
    labels_names = [label.name for label in labels]
    nb_labels = len(labels_names)
    histories = []
    mean_loss = 0
    mean_accuracy = 0
    total_conf_matrix = np.zeros((len(labels_names), len(labels_names)))

    model.summary()

    model_checkpoint_cb = None
    if with_model_checkpoint:
        # Model checkpoint callback should be created here to avoid resetting the 'best' property of the model checkpoint.
        # By doing so, we ensure that model checkpoint is the best across all cross validations.
        model_file = MODEL_PATH + model_name + ".hdf5"
        model_checkpoint_cb = ModelCheckpoint(model_file, monitor='f1_score_val', verbose=0, save_best_only=True, mode='max')

# for validation in range(nb_cross_validations):
#     fold = 0
#
#     skcv = spacv.SKCV(n_splits=5, buffer_radius=0.1).split(gpd_df)
#
#     for train, val in skcv:
#         train_set = [img for i, img in enumerate(dataset) if i in train]
#         validation_set = [img for i, img in enumerate(dataset) if i in val]
#
#         X_train = images_from_dataset(train_set, bands)
#         y_train = labels_from_dataset(train_set, labels_names)
#         Y_train = to_categorical(y_train, num_classes=nb_labels)
#
#         X_validation = images_from_dataset(validation_set, bands)
#         y_validation = labels_from_dataset(validation_set, labels_names)
#         Y_validation = to_categorical(y_validation, num_classes=nb_labels)
#
#         history, trained_model = train_fold(
#             X_validation=X_validation,
#             X_train=X_train,
#             Y_validation=Y_validation,
#             Y_train=Y_train,
#             y_validation=y_validation,
#             y_train=y_train,
#             epochs=epochs,
#             fold=fold,
#             validation=validation,
#             model=model,
#             early_stopping=early_stopping,
#             model_checkpoint_cb=model_checkpoint_cb,
#             new_model=True,
#         )
#
#         conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation,
#                                                      nb_labels)
#
#         fold_size = len(y_train)
#
#         histories.append(history)
#         total_conf_matrix += conf_matrix
#         mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
#         mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)
#
#         fold += 1
#
# return mean_loss, mean_accuracy, histories, total_conf_matrix


def display_cross_val_map_class(fold_sets, map_shape, title, legends=["Other", "Test"], xlim=[106,110], ylim=[10, 16], figsize=(12, 6)):
    """
    code adapted from Romain Capocasale Master thesis display_cross_val_map_class
    :param sets
    :param map_shape:
    :param title:
    :param legend1:
    :param legend2:
    :param xlim:
    :param ylim:
    :param figsize:
    :return:
    """

    fig, ax = pl.subplots(figsize=figsize)
    map_shape.plot(ax=ax, facecolor='Grey', edgecolor='k', alpha=0.5, linewidth=0.3)

    ax.set_prop_cycle(pl.cycler('color', ['tab:orange', 'tab:green', 'tab:red', 'tab:blue', 'tab:purple', 'tab:brown']))

    for fold_set in fold_sets:
        fold_set.plot(ax=ax, markersize=1, categorical=True, legend=True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")

    legend = ax.legend(legends)

    for handle in legend.legendHandles:
        handle.set_sizes([30])

    fig.suptitle(title)



