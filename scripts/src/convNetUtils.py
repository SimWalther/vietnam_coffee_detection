from math import floor
from sklearn.utils import class_weight
from sklearn import metrics as me
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from shapely import wkt
import numpy as np
import pandas as pd
import geopandas as gpd
from config import *
import spacv
from tifffile.tifffile import imread
from rasterio.plot import reshape_as_image
from rasterUtils import square_chunks
from sklearn.model_selection import StratifiedKFold, KFold

from labelsUtils import categories_from_label_set
from lossesUtils import categorical_focal_loss
from visualizationUtils import display_cross_val_map_class
from albumentations import (
    Compose,
    HorizontalFlip,
    ShiftScaleRotate,
    VerticalFlip,
    RandomRotate90,
)

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


class ImageSequence(Sequence):
    """
    Code inspired by https://medium.com/the-artificial-impostor/custom-image-augmentation-with-keras-70595b01aeac,
    and https://github.com/keras-team/keras/issues/9707
    """

    def __init__(self, x_set, y_set, batch_size, augmentations):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.augment = augmentations
        self.batch_indices = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # NOTE: It have image paths instead of rasters there but it would slow down the processing
        # it would be an alternative if memory was constrained

        indices = self.batch_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indices]
        batch_y = self.y[indices]

        return np.stack(
            [self.augment(image=image)["image"] for image in batch_x],
            axis=0
        ), np.asarray(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.batch_indices)


class ImageMultiOutputSequence(Sequence):
    """
    Code inspired by https://medium.com/the-artificial-impostor/custom-image-augmentation-with-keras-70595b01aeac,
    and https://github.com/keras-team/keras/issues/9707
    """

    def __init__(self, x_set, y_sets, output_names, batch_size, augmentations):
        assert len(output_names) == len(y_sets), "You must have as many output sets as output names"
        self.x = x_set
        self.y = y_sets
        self.output_names = output_names
        self.batch_size = batch_size
        self.augment = augmentations
        self.batch_indices = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        indices = self.batch_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[indices]
        batch_y = {
            output_name: self.y[i][indices] for i, output_name in enumerate(self.output_names)
        }

        return np.stack(
            [self.augment(image=image)["image"] for image in batch_x],
            axis=0
        ), batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.batch_indices)


class Metrics(Callback):
    """
    Keras callback to provides additional metrics.
    It logs f1-score of the train and validation after each epoch. Those metrics can be used by the early stopping.
    Code inspired by: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    """

    def __init__(self, train_datagen=None, validation_datagen=None):
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
        # Warning: This will load every training data in memory

        if self.train_datagen is not None:
            train_images, train_classes = elements_inside_data_generator(self.train_datagen)

            if isinstance(self.train_datagen, ImageMultiOutputSequence):
                target_train = [np.argmax(train_classes[output_name], axis=-1) for output_name in train_classes.keys()]
                pred = self.model.predict(train_images)
                predicted_train = [np.argmax(pred[i], axis=-1) for i in range(len(pred))]

                # Here we define f1-score of the multi-output model as the mean of f1-score of each output
                f1_score_train = np.mean([me.f1_score(target_train[i], predicted_train[i], average="macro") for i in range(len(predicted_train))])
            else:
                target_train = np.argmax(train_classes, axis=-1)
                predicted_train = np.argmax(np.asarray(self.model.predict(train_images)), axis=-1)
                f1_score_train = me.f1_score(target_train, predicted_train, average="macro")

            logs['f1_score_train'] = f1_score_train

        if self.validation_datagen is not None:
            validation_images, validation_classes = elements_inside_data_generator(self.validation_datagen)

            if isinstance(self.validation_datagen, ImageMultiOutputSequence):
                target_validation = [np.argmax(validation_classes[output_name], axis=-1) for output_name in validation_classes.keys()]
                pred = self.model.predict(validation_images)
                predicted_validation = [np.argmax(pred[i], axis=-1) for i in range(len(pred))]

                # Here we define f1-score of the multi-output model as the mean of f1-score of each output
                f1_score_val = np.mean([me.f1_score(target_validation[i], predicted_validation[i], average="macro") for i in range(len(predicted_validation))])
            else:
                target_validation = np.argmax(validation_classes, axis=-1)
                predicted_validation = np.argmax(np.asarray(self.model.predict(validation_images)), axis=-1)
                f1_score_val = me.f1_score(target_validation, predicted_validation, average="macro")

            logs['f1_score_val'] = f1_score_val

        return


def elements_inside_data_generator(datagen):
    images = []

    if isinstance(datagen.__getitem__(0)[1], dict):
        true_classes = {output_name: [] for output_name in datagen.__getitem__(0)[1].keys()}
    else:
        true_classes = []

    for i in range(len(datagen)):
        images.extend(datagen.__getitem__(i)[0])

        if isinstance(datagen.__getitem__(i)[1], dict):
            for output_name, value in datagen.__getitem__(i)[1].items():
                true_classes[output_name].extend(value.tolist())
        else:
            true_classes.extend(datagen.__getitem__(i)[1].tolist())

    if not isinstance(datagen.__getitem__(i)[1], dict):
        true_classes = np.asarray(true_classes)

    return np.asarray(images), true_classes


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
                early_stopping=False, model_checkpoint_cb=None, categories=None):
    """
    Train a Keras neural network model
    :param categories: categories used
    :param model_checkpoint_cb: the checkpoint callback
    :param model: the Keras neural network model
    :param train_datagen train data generator
    :param validation_datagen validation data generator
    :param class_weights: weight of each class. If classes are imbalanced it will gives more importance
    to underrepresented classes
    :param epochs: the number of epochs
    :param steps_per_epoch: the number of steps per epoch
    :param early_stopping: defines if early stopping is used
    :return: the train history and the trained model
    """

    callbacks = []

    callbacks.append(Metrics(train_datagen=train_datagen, validation_datagen=validation_datagen))

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

    history = model.fit(**fit_args)
    return history, model


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


def evaluate_multi_output_model(model, X_test, Y_test, y_test):
    # Evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)
    loss = score[0]
    accuracy = score[1]

    # Predict labels on batch
    pred = model.predict_on_batch(X_test)
    pred = [np.argmax(pred[i], axis=-1) for i in range(len(pred))]

    # Confusion matrix
    conf_matrices = [me.confusion_matrix(y_test[i], pred[i], labels=np.unique(y_test[i])) for i in range(len(pred))]

    return conf_matrices, accuracy, loss


def compute_class_weights(y_train):
    # Compute each classes weight
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(enumerate(class_weights))


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
        model_file = os.path.join(MODEL_ROOT_PATH, model_name + ".hdf5")
        model_checkpoint_cb = ModelCheckpoint(
            model_file, monitor='f1_score_val',
            verbose=0, save_best_only=True, mode='max'
        )

    images = images_from_dataset(dataset, bands)
    true_classes = labels_from_dataset(dataset, labels_names)

    for nth_cross_validation in range(nb_cross_validations):
        fold = 0

        for train_index, validation_index in StratifiedKFold(n_splits=k, shuffle=True).split(images, true_classes):
            X_train, X_validation = images[train_index], images[validation_index]
            y_train, y_validation = true_classes[train_index], true_classes[validation_index]
            Y_train = to_categorical(y_train, num_classes=len(labels_names))
            Y_validation = to_categorical(y_validation, num_classes=len(labels_names))
            fold_size = len(y_train)

            # create data generators
            train_datagen = ImageSequence(
                X_train, Y_train, batch_size=32, augmentations=AUGMENTATIONS
            )
            validation_datagen = ImageSequence(
                X_validation, Y_validation, batch_size=32,
                augmentations=VALIDATION_AUGMENTATIONS
            )
            class_weights = compute_class_weights(y_train)

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
                steps_per_epoch=fold_size / 32,
                early_stopping=early_stopping,
                model_checkpoint_cb=model_checkpoint_cb,
            )

            conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation, nb_labels)

            histories.append(history)
            total_conf_matrix += conf_matrix
            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

            fold += 1

    return mean_loss, mean_accuracy, histories, total_conf_matrix


def cross_validation_multi_output_model(model, dataset, bands, labels, epochs, nb_cross_validations=1, k=5, early_stopping=False,
                                        with_model_checkpoint=False, model_name="model", categories=None):
    """
    :param categories: categories used
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
    total_conf_matrices = [np.zeros((len(labels_names), len(labels_names))), np.zeros((len(categories), len(categories)))]

    model.summary()

    model_checkpoint_cb = None

    if with_model_checkpoint:
        # Model checkpoint callback should be created here to avoid resetting the 'best' property of the model checkpoint.
        # By doing so, we ensure that model checkpoint is the best across all cross validations.
        model_file = os.path.join(MODEL_ROOT_PATH, model_name + ".hdf5")
        model_checkpoint_cb = ModelCheckpoint(
            model_file, monitor='f1_score_val',
            verbose=0, save_best_only=True, mode='max'
        )

    images = images_from_dataset(dataset, bands)
    true_classes = labels_from_dataset(dataset, labels_names)
    categories_train = categories_from_label_set(labels, true_classes)

    for nth_cross_validation in range(nb_cross_validations):
        fold = 0

        for train_index, validation_index in KFold(n_splits=k, shuffle=True).split(images, true_classes):
            X_train, X_validation = images[train_index], images[validation_index]
            y_train, y_validation = [true_classes[train_index], categories_train[train_index]], \
                                    [true_classes[validation_index], categories_train[validation_index]]
            Y_train = [
                to_categorical(y_train[0], num_classes=len(labels_names)),
                to_categorical(y_train[1], num_classes=len(categories)),
            ]
            Y_validation = [
                to_categorical(y_validation[0], num_classes=len(labels_names)),
                to_categorical(y_validation[1], num_classes=len(categories)),
            ]
            fold_size = len(y_train[0])

            # create data generators
            train_datagen = ImageMultiOutputSequence(
                X_train, Y_train, ['label', 'category'], batch_size=32, augmentations=AUGMENTATIONS
            )
            validation_datagen = ImageMultiOutputSequence(
                X_validation, Y_validation, ['label', 'category'], batch_size=32, augmentations=VALIDATION_AUGMENTATIONS
            )
            class_weights = None  # class weights cannot be used with multi output model

            print(f"\nValidation {nth_cross_validation + 1}, fold {fold + 1} :\n---------------------------\n")

            # clone given model without keeping the layers weights
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(optimizer='adam', loss={
                'label': categorical_focal_loss([[.25] * len(labels_names)]),
                'category': categorical_focal_loss([[.25] * len(categories)])
            }, metrics={
                'label': 'accuracy',
                'category': 'accuracy'
            })

            history, trained_model = train_model(
                model=current_model,
                train_datagen=train_datagen,
                validation_datagen=validation_datagen,
                class_weights=class_weights,
                epochs=epochs,
                steps_per_epoch=fold_size / 32,
                early_stopping=early_stopping,
                model_checkpoint_cb=model_checkpoint_cb,
                categories=categories
            )

            conf_matrices, accuracy, loss = evaluate_multi_output_model(trained_model, X_validation, Y_validation, y_validation)
            histories.append(history)

            for i, conf_matrix in enumerate(conf_matrices):
                total_conf_matrices[i] += conf_matrix

            mean_accuracy += accuracy * fold_size / (len(dataset) * nb_cross_validations)
            mean_loss += loss * fold_size / (len(dataset) * nb_cross_validations)

            fold += 1

    return mean_loss, mean_accuracy, histories, total_conf_matrices


def cross_validation_from_csv_files(model, other_filename, test_filename, bands, labels, epochs,
                                    nb_cross_validations=1, early_stopping=False, with_model_checkpoint=False,
                                    model_name="model"):
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
        model_file = os.path.join(MODEL_ROOT_PATH, model_name + ".hdf5")
        model_checkpoint_cb = ModelCheckpoint(model_file, monitor='f1_score_val', verbose=0, save_best_only=True,
                                              mode='max')

    for nth_cross_validation in range(nb_cross_validations):
        fold = 0

        for validation, train in spatial_separation_dataset(other_df, labels):
            display_cross_val_map_class([train, validation, test_df], vietnam_shape, f"Cross validation split fold",
                                        legends=["Train", "Validation", "Test"])

            X_train = np.asarray([
                prepare_image(img, bands) for img in train['images'].to_numpy()
            ])
            X_validation = np.asarray([
                prepare_image(img, bands) for img in validation['images'].to_numpy()
            ])
            y_train = np.array([labels_names.index(label) for label in train['label'].to_numpy()])
            y_validation = np.array([labels_names.index(label) for label in validation['label'].to_numpy()])
            Y_train = to_categorical(y_train, num_classes=len(labels_names))
            Y_validation = to_categorical(y_validation, num_classes=len(labels_names))

            # create data generators
            batch_size = 32
            train_datagen = ImageSequence(X_train, Y_train, batch_size, AUGMENTATIONS)
            validation_datagen = ImageSequence(X_validation, Y_validation, batch_size, VALIDATION_AUGMENTATIONS)

            class_weights = compute_class_weights(y_train)

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

            conf_matrix, accuracy, loss = evaluate_model(trained_model, X_validation, Y_validation, y_validation,
                                                         nb_labels)

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

    images = images_from_dataset(dataset, bands)
    true_classes = labels_from_dataset(dataset, labels_names)

    for nth_cross_validation in range(nb_cross_validations):
        fold = 0

        for train_index, validation_index in StratifiedKFold(n_splits=k, shuffle=True).split(images, true_classes):
            X_train, X_validation = images[train_index], images[validation_index]
            y_train, y_validation = true_classes[train_index], true_classes[validation_index]
            Y_train = to_categorical(y_train, num_classes=len(labels_names))
            Y_validation = to_categorical(y_validation, num_classes=len(labels_names))

            # We must reinitialize the model each fold!
            current_model = clone_model(model)

            # Specify optimizer and loss function
            current_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            for metric in range(nb_metrics):
                # create data generators
                batch_size = 32
                train_datagen = ImageSequence(X_train, Y_train, batch_size, AUGMENTATIONS)
                validation_datagen = ImageSequence(X_validation, Y_validation, batch_size, VALIDATION_AUGMENTATIONS)

                class_weights = compute_class_weights(y_train)

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

            fold += 1

    return mean_losses, mean_accuracies, histories, total_conf_matrices


def prepare_image(image, bands):
    # if this is a string we assume it's a path the image
    # otherwise we process image as rasters
    if isinstance(image, str):
        image = imread(image)

    # filter bands
    image = np.asarray([image[band] for i, band in enumerate(bands)])
    return reshape_as_image(image)


def images_from_dataset(dataset, bands):
    """
    Extract images from dataset
    :param dataset: the dataset
    :return: an images array
    """
    return np.asarray([
        prepare_image(data[1], bands) for data in dataset
    ])


def labels_from_dataset(dataset, labels_names):
    """
    Extract labels from a dataset into a numpy array
    :param dataset: the dataset
    :param labels_names: the names of the labels to keep
    :return: a labels array
    """
    return np.array([labels_names.index(img[0]) for img in dataset])


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
        current_selected_fold = [i, i + 1]
        current_other = np.concatenate([np.arange(0, i), np.arange(i + 2, nb_fold)])

        selected = pd.DataFrame()
        other = pd.DataFrame()

        for label in range(len(labels_names)):
            selected = selected.append(
                labels_data[label].iloc[
                    np.concatenate([list(fold_list[fold][label]) for fold in current_selected_fold]).ravel().tolist()]
            )

            other = other.append(
                labels_data[label].iloc[
                    np.concatenate([list(fold_list[fold][label]) for fold in current_other]).ravel().tolist()]
            )

        # shuffle rows
        selected = selected.sample(frac=1).reset_index(drop=True)
        other = other.sample(frac=1).reset_index(drop=True)

        yield selected, other


# find all raster files in the folder and create a dataset with them
def predict_on_raster(trained_model, raster_path, bands, square_size=9):
    predictions = []
    image_indices = []

    for batch_images, batch_indices in square_chunks(raster_path, square_size):
        images = np.asarray([
            prepare_image(img, bands) for img in batch_images
        ])

        pred = trained_model.predict(images)
        pred = np.argmax(pred, axis=-1)

        predictions.extend(pred)
        image_indices.extend(batch_indices)

    return predictions, image_indices


def predict_label_category_on_raster(trained_model, raster_path, bands, square_size=9):
    label_predictions = []
    category_predictions = []
    image_indices = []

    for batch_images, batch_indices in square_chunks(raster_path, square_size):
        images = np.asarray([
            prepare_image(img, bands) for img in batch_images
        ])

        pred = trained_model.predict(images)
        label_pred = np.argmax(pred[0], axis=-1)
        category_pred = np.argmax(pred[1], axis=-1)

        label_predictions.extend(label_pred)
        category_predictions.extend(category_pred)
        image_indices.extend(batch_indices)

    return label_predictions, category_predictions, image_indices
