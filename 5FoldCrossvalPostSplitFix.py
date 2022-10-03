import io
from contextlib import redirect_stdout

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedGroupKFold

import io
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from custom_utils import formatDf, plot_to_image

jointsNoUnk = formatDf(
    "/home/msa-project/Dataset_joints/data_apollo/joint_annotations.csv")

# Input Images Variables
TARGET_SIZE = (256,256)
INPUT_SHAPE = TARGET_SIZE + (1,)
RESCALE_FACTOR = 1.0/255

# Architechture Varibales
NUM_CLASSES = 4
CONV_LAYERS = 5 # starting first layer with 32 filters, and each sequential doubling the number of filters
DENSE_UNITS = 256

# Processing Variables
BATCH_SIZE = 32
EPOCHS = 100

def create_classifier(num_classes, input_shape, conv_layers, dense_units):

    # Create CNN with general arch of:
    # Sequential Convolutional-ReLu-MaxPooling Layers of Doubling Num of Filters
    # A Pair of Dense Layers at the end of the convolutional sequence
    # An Output Layer with Nodes = num_classes

    # Restrictions on function's parameters:
    ## num_classes >= 2
    ## conv_layers >= 1

    classifier = Sequential()

    # Initial Convolutional Block
    classifier.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        input_shape=input_shape,
        activation="relu")
    )
    classifier.add(MaxPooling2D(
        pool_size=(2, 2)
    ))
    classifier.add(Dropout(
        rate=0.2))

    # Additional Convolutional Block with twice the num of filters as previous block
    for cl in range(conv_layers-1):
        # when cl = 0, power = 6, so 2^6 = 64 (since initial layer has 32 filters)
        power = 6 + cl

        classifier.add(Conv2D(
            filters=2 ** power,
            kernel_size=(3, 3),
            padding="same",
            activation="relu")
        )
        classifier.add(MaxPooling2D(
            pool_size=(2, 2)
        ))
        classifier.add(Dropout(
            rate=0.2))

    classifier.add(Flatten())

    # Dense Layers
    classifier.add(Dense(
        units=dense_units,
        activation="relu"
    ))
    classifier.add(Dropout(
        rate=0.5))

    # Output Layer
    activation = "softmax"  # default activation for multi-class classification
    if num_classes == 2:
        activation == "sigmoid"  # change activation in case of binary classification

    classifier.add(Dense(
        units=num_classes,
        activation=activation
    ))

    return classifier

k_results = []
fold = 0
arch_write_flag = True

cv = StratifiedGroupKFold(n_splits=5)
cv1 = StratifiedGroupKFold(n_splits=4)
for learn_idxs, test_idxs in cv.split(jointsNoUnk["Filename"], jointsNoUnk["ml_class"], jointsNoUnk["patient"]):
    fold = fold+1
    
    learn = jointsNoUnk.iloc[learn_idxs]
    test = jointsNoUnk.iloc[test_idxs]

    train_idxs, val_idxs = next(cv1.split(learn["Filename"], learn["ml_class"], learn["patient"]))
    train = learn.iloc[train_idxs]
    val = learn.iloc[val_idxs]
 

    classifier = create_classifier(num_classes=NUM_CLASSES,input_shape=INPUT_SHAPE,conv_layers=CONV_LAYERS,dense_units=DENSE_UNITS)

    # (Re-)Compile Model
    classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

    gen_train = ImageDataGenerator(
        rescale=RESCALE_FACTOR,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    gen_val = ImageDataGenerator(rescale=RESCALE_FACTOR)
    gen_test = ImageDataGenerator(rescale = RESCALE_FACTOR)

    set_train = gen_train.flow_from_dataframe(
        train,
        directory="/home/msa-project/cropped",
        x_col='Filename',
        y_col='ml_class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale"
    )

    set_val = gen_val.flow_from_dataframe(
        val,
        directory="/home/msa-project/cropped",
        x_col='Filename',
        y_col='ml_class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale"
    )

    set_test = gen_test.flow_from_dataframe(
        test,
        directory="/home/msa-project/cropped",
        x_col='Filename',
        y_col='ml_class',
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale",
        shuffle = False
    )

    log_dir = 'tensorboard_logs/postSplitFixDense256/fold' + str(fold)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    if(arch_write_flag):
        with io.StringIO() as buf, redirect_stdout(buf):
            classifier.summary()
            output = buf.getvalue()

            file_writer1 = tf.summary.create_file_writer(log_dir)
            with file_writer1.as_default():
                with tf.name_scope("Architecture"):
                    tf.summary.text("Arch",output,step=fold)
        arch_write_flag=False

    # Fit Kth Model
    fitted_model = classifier.fit(
        set_train,
        epochs=EPOCHS,
        validation_data=set_val,
        callbacks=[tensorboard_callback]
    )

    results = classifier.evaluate(set_test)
    result_text = "Loss: "+str(results[0])+" Accuracy: " + str(results[1])
    file_writer = tf.summary.create_file_writer(log_dir)
    with file_writer.as_default():
        with tf.name_scope("Test_Metrics"):
            tf.summary.text("Fold"+str(fold),result_text,step=fold)

    set_test.reset()
    y_pred = classifier.predict(set_test, set_test.n // BATCH_SIZE+1)
    class_pred = np.argmax(y_pred, axis=1)
    labels = set_test.class_indices
    cm = confusion_matrix(set_test.classes, class_pred)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = list(labels.keys()))
    cm_disp.plot()
    cm_image = plot_to_image(plt.gcf())

    file_writer2 = tf.summary.create_file_writer(log_dir)
    with file_writer2.as_default():
        with tf.name_scope("Test_Confusion_Matrix"):
            tf.summary.image("Fold"+str(fold), cm_image,step=fold)