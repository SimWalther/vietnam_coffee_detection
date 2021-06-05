from tensorflow.keras.utils import Sequence
from math import floor
import matplotlib.pyplot as pl
from sklearn.utils import class_weight
from sklearn import metrics as me
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from shapely import wkt
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import spacv
from libtiff import TIFF
from rasterio.plot import reshape_as_image

from albumentations import (
    Compose,
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    RandomRotate90,
)

DATA_ROOT_PATH = '../data/'
MODEL_PATH = '../models/'
DATASET_PATH = DATA_ROOT_PATH + 'datasets/'
DISTRICTS_PATH = DATA_ROOT_PATH + "districts/diaphantinh.geojson"

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

# We must not apply augmentations on validation data :
# https://datascience.stackexchange.com/questions/41422/when-using-data-augmentation-is-it-ok-to-validate-only-with-the-original-images
VALIDATION_AUGMENTATIONS = Compose([])


# FIXME: shuffle data generators
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

        return np.stack(
            [self.augment(image=image)["image"] for image in batch_x],
            axis=0
        ), np.array(batch_y)


class Metrics(Callback):
    """
    Keras callback to provides additional metrics.
    It logs f1-score of the train and validation after each epoch. Those metrics can be used by the early stopping.
    Code inspired by: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    """

    def __init__(self, train_datagen, validation_datagen):
        """
        Initialize callback
        :param train: the train generator
        :param validation: the validation generator
        """

        super(Metrics, self).__init__()
        self._supports_tf_logs = True
        self.train_datagen = train_datagen
        self.validation_datagen = validation_datagen

    def on_epoch_end(self, epoch, logs={}):
        # This will load every training data in memory
        train_images, train_classes = elements_inside_data_generator(self.train_datagen)
        target_train = np.argmax(train_classes, axis=-1)
        predicted_train = np.argmax(np.asarray(self.model.predict(train_images)), axis=-1)
        f1_score_train = me.f1_score(target_train, predicted_train, average="macro")

        validation_images, validation_classes = elements_inside_data_generator(self.train_datagen)
        target_validation = np.argmax(validation_classes, axis=-1)
        predicted_validation = np.argmax(np.asarray(self.model.predict(validation_images)), axis=-1)
        f1_score_val = me.f1_score(target_validation, predicted_validation, average="macro")

        logs['f1_score_train'] = f1_score_train
        logs['f1_score_val'] = f1_score_val

        return


def elements_inside_data_generator(datagen):
    images = []
    true_classes = []

    for i in range(len(datagen)):
        images.extend(datagen.__getitem__(i)[0])
        true_classes.extend(datagen.__getitem__(i)[1].tolist())

    return np.asarray(images), np.asarray(true_classes)


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


def train_model(model, train_datagen, validation_datagen, class_weights, epochs, steps_per_epoch,
                early_stopping=False, model_checkpoint_cb=None):
    """
    Train a Keras neural network model
    :param model: the Keras neural network model
    :param train_datagen train data generator
    :param validation_datagen validation data generator
    :param class_weights: weight of each class. If classes are imbalanced it will gives more importance
    to underrepresented classes
    :param epochs: the number of epochs
    :param steps_per_epoch: the number of steps per epoch
    :param early_stopping: defines if early stopping is used
    :return: the train history
    """

    callbacks = [Metrics(train_datagen=train_datagen, validation_datagen=validation_datagen)]

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
        validation_data=validation_datagen,
        callbacks=callbacks
    )

    return model.fit(**fit_args)


def separate_data_into_images_and_labels(data, labels_names, bands):
    images = images_from_dataset(data, bands)
    labels = labels_from_dataset(data, labels_names)
    one_hot_labels = to_categorical(labels, num_classes=len(labels_names))

    return images, labels, one_hot_labels


def split_fold_into_train_validation_sets(dataset, fold_start, fold_end, labels_names, bands):
    """
    Get train and validation sets from a given fold
    :param dataset: the dataset
    :param fold_start: start position of the fold in the dataset
    :param fold_end: end position of the fold in the dataset
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

    X_train, y_train, Y_train = separate_data_into_images_and_labels(train, labels_names, bands)
    X_validation, y_validation, Y_validation = separate_data_into_images_and_labels(validation, labels_names, bands)

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

    for nth_cross_validation in range(nb_cross_validations):
        np.random.shuffle(dataset)

        for fold, fold_indices in enumerate(k_fold_indices(dataset, k)):
            X_train, y_train, Y_train, X_validation, y_validation, Y_validation = split_fold_into_train_validation_sets(
                dataset, fold_indices[0], fold_indices[1], labels_names, bands
            )

            # create data generators
            batch_size = 32
            train_datagen = LandsatSequence(X_train, Y_train, batch_size, AUGMENTATIONS)
            validation_datagen = LandsatSequence(X_validation, Y_validation, batch_size, VALIDATION_AUGMENTATIONS)

            # Compute each classes weight
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )

            class_weights = dict(enumerate(class_weights))

            print(f"\nValidation {nth_cross_validation + 1}, fold {fold + 1} :\n---------------------------\n")

            # clone given model without keeping the layers weights
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history, trained_model = train_model(
                model=current_model,
                train_datagen=train_datagen,
                validation_datagen=validation_datagen,
                class_weights=class_weights,
                epochs=epochs,
                steps_per_epoch=len(y_train) / 32,
                early_stopping=early_stopping,
                model_checkpoint_cb=model_checkpoint_cb,
            )

            conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation, nb_labels)

            fold_size = len(y_train)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_loss, mean_accuracy, histories, total_conf_matrix


def cross_validation_from_csv_files(model, other_filename, test_filename, bands, labels, epochs,
                                     nb_cross_validations=1, early_stopping=False, with_model_checkpoint=False, model_name="model"):
    labels_names = [label.name for label in labels]
    nb_labels = len(labels_names)
    histories = []
    mean_loss = 0
    mean_accuracy = 0
    total_conf_matrix = np.zeros((len(labels_names), len(labels_names)))

    # Load the vietnam shape used to draw the map
    vietnam_shape = gpd.read_file(DISTRICTS_PATH)

    # Load csv into geodataframes
    test_df = gpd.GeoDataFrame(pd.read_csv(os.path.join(DATASET_PATH, test_filename)))
    test_df['geometry'] = test_df['geometry'].apply(wkt.loads)
    other_df = gpd.GeoDataFrame(pd.read_csv(os.path.join(DATASET_PATH, other_filename)))
    other_df['geometry'] = other_df['geometry'].apply(wkt.loads)

    nb_images = len(test_df) + len(other_df)

    model.summary()

    model_checkpoint_cb = None

    if with_model_checkpoint:
        # Model checkpoint callback should be created here to avoid resetting the 'best' property of the model checkpoint.
        # By doing so, we ensure that model checkpoint is the best across all cross validations.
        model_file = MODEL_PATH + model_name + ".hdf5"
        model_checkpoint_cb = ModelCheckpoint(model_file, monitor='f1_score_val', verbose=0, save_best_only=True, mode='max')

    for nth_cross_validation in range(nb_cross_validations):
        fold = 0

        for validation, train in spatial_separation_dataset(other_df, labels):
            # display_cross_val_map_class([train, validation, test_df], vietnam_shape, f"Cross validation split fold", ["Train", "Validation", "Test"])
            X_train = prepare_images(train['images'].to_numpy(), bands)
            X_validation = prepare_images(validation['images'].to_numpy(), bands)
            y_train = np.array([labels_names.index(label) for label in train['label'].to_numpy()])
            y_validation = np.array([labels_names.index(label) for label in validation['label'].to_numpy()])
            Y_train = to_categorical(y_train, num_classes=len(labels_names))
            Y_validation = to_categorical(y_validation, num_classes=len(labels_names))

            # create data generators
            batch_size = 32
            train_datagen = LandsatSequence(X_train, Y_train, batch_size, AUGMENTATIONS)
            validation_datagen = LandsatSequence(X_validation, Y_validation, batch_size, VALIDATION_AUGMENTATIONS)

            # Compute each classes weight
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )

            class_weights = dict(enumerate(class_weights))

            print(f"\nValidation {nth_cross_validation + 1}, fold {fold + 1} :\n---------------------------\n")

            # clone given model without keeping the layers weights
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history, trained_model = train_model(
                model=current_model,
                train_datagen=train_datagen,
                validation_datagen=validation_datagen,
                class_weights=class_weights,
                epochs=epochs,
                steps_per_epoch=len(y_train) / 32,
                early_stopping=early_stopping,
                model_checkpoint_cb=model_checkpoint_cb,
            )

            conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation, nb_labels)

            fold_size = len(y_train)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (nb_images * nb_cross_validations)
            mean_loss += loss * fold_size / (nb_images * nb_cross_validations)

            fold += 1

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

    for nth_cross_validation in range(nb_cross_validations):
        np.random.shuffle(dataset)

        for fold, fold_indices in enumerate(k_fold_indices(dataset, k)):
            X_train, y_train, Y_train, X_validation, y_validation, Y_validation = split_fold_into_train_validation_sets(
                dataset, fold_indices[0], fold_indices[1], labels_names, bands
            )

            # We must reinitialize the model each fold!
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            for metric in range(nb_metrics):
                # create data generators
                batch_size = 32
                train_datagen = LandsatSequence(X_train, Y_train, batch_size, AUGMENTATIONS)
                validation_datagen = LandsatSequence(X_validation, Y_validation, batch_size, VALIDATION_AUGMENTATIONS)

                # Compute each classes weight
                class_weights = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_train),
                    y=y_train
                )

                class_weights = dict(enumerate(class_weights))

                print(f"\nValidation {nth_cross_validation + 1}, fold {fold + 1} :\n---------------------------\n")

                history, trained_model = train_model(
                    model=current_model,  # reuse the same model
                    train_datagen=train_datagen,
                    validation_datagen=validation_datagen,
                    class_weights=class_weights,
                    epochs=epochs,
                    steps_per_epoch=len(y_train) / 32,
                    early_stopping=False,
                    model_checkpoint_cb=False,
                )

                conf_matrix, accuracy, loss = evaluate_model(current_model, X_validation, Y_validation, y_validation,
                                                             nb_labels)

                fold_size = len(y_train)

                histories.append(history)
                total_conf_matrices[metric] += conf_matrix
                mean_accuracies[metric] += accuracy * fold_size / (len(dataset) * nb_cross_validations)
                mean_losses[metric] += loss * fold_size / (len(dataset) * nb_cross_validations)

    return mean_losses, mean_accuracies, histories, total_conf_matrices


def prepare_images(images, bands):
    """
    Prepare images to have the right format and contains only bands we will use
    Images format is converted from (bands, rows, columns) to (rows, columns, bands)
    :param images: the images
    :param bands: the bands to keep
    :return: an images array
    """
    # if this is a string we assume it's a path the image
    # otherwise we process images as rasters
    if isinstance(images[0], str):
        images = np.array([TIFF.open(path, mode='r').read_image() for path in images])
    else:
        images = np.array([img for img in images])

    # Filter bands
    images = [
        [
            img[band] for i, band in enumerate(bands)
        ] for img in images
    ]

    return np.array([reshape_as_image(img) for img in images])


def images_from_dataset(dataset, bands):
    """
    Extract images from dataset
    :param dataset: the dataset
    :return: an images array
    """
    return prepare_images(np.array([img[1] for img in dataset]), bands)


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
        label_data = geo_df[geo_df['label'] == label_name]
        labels_data.append(label_data)

        fold = 0

        for _, validation in spacv.SKCV(n_splits=nb_fold, buffer_radius=0.1).split(label_data):
            fold_list[fold][label_index] = validation

            fold += 1

    # We know that each validation set is unique
    # we choose two validation sets for the test set
    # and we keep other for the train and validation
    nb_grouped_fold = 2

    np.random.shuffle(fold_list)

    for i in range(0, nb_fold, nb_grouped_fold):
        current_selected_fold = [i, i+1]
        current_other = np.concatenate([np.arange(0, i), np.arange(i+2, nb_fold)])

        selected = pd.DataFrame()
        other = pd.DataFrame()

        for label in range(len(labels_names)):
            selected = selected.append(
                labels_data[label].iloc[np.concatenate([list(fold_list[fold][label]) for fold in current_selected_fold]).ravel().tolist()]
            )

            other = other.append(
                labels_data[label].iloc[np.concatenate([list(fold_list[fold][label]) for fold in current_other]).ravel().tolist()]
            )

        # shuffle rows
        selected = selected.sample(frac=1).reset_index(drop=True)
        other = other.sample(frac=1).reset_index(drop=True)

        yield selected, other


def display_cross_val_map_class(fold_sets, map_shape, title, legends=["Other", "Test"], xlim=[106,110], ylim=[10, 16], figsize=(12, 6)):
    """
    code adapted from Romain Capocasale Master thesis display_cross_val_map_class
    :param fold_sets:
    :param map_shape:
    :param title:
    :param legends:
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



