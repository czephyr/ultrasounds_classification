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

from custom_utils import formatKneeDf, plot_to_image

dfNoOther = formatKneeDf(
    "/home/msa-project/Dataset_joints/data_apollo/knee_annotations.csv")


# Input Images Variables
TARGET_SIZE = (256,256)
INPUT_SHAPE = TARGET_SIZE + (1,)
RESCALE_FACTOR = 1.0/255

# Architechture Varibales
NUM_CLASSES = 4
DENSE_UNITS = 128
NUM_TOTAL_CONVS = 4

# Processing Variables
BATCH_SIZE = 32
EPOCHS = 200

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
num = 0
arch_write_flag = True

for i in [0.8,0.6,0.4,0.2,0.1]:
    cv = StratifiedGroupKFold(n_splits=5)
    cv1 = StratifiedGroupKFold(n_splits=4)
    learn_idxs, test_idxs = next(cv.split(dfNoOther["Filename"], dfNoOther["ml_class"], dfNoOther["patient"]))
    learn = dfNoOther.iloc[learn_idxs]
    test = dfNoOther.iloc[test_idxs]

    print("LEARN:"+str(len(learn)))
    print("TEST:"+str(len(test)))

    train_idxs, val_idxs = next(cv1.split(learn["Filename"], learn["ml_class"], learn["patient"]))
    train = learn.iloc[train_idxs]
    val = learn.iloc[val_idxs]

    train = train.sample(frac=i)
    print("TRAIN:"+str(len(train)))
    print("VAL:"+str(len(val)))
    print("--")

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

    classifier = create_classifier(num_classes=NUM_CLASSES,input_shape=INPUT_SHAPE,conv_layers=NUM_TOTAL_CONVS,dense_units=DENSE_UNITS)
    classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

    log_dir = 'tensorboard_logsPart2/tossExpl'+str(i)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

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
            tf.summary.text("num"+str(i),result_text,step=num)

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
            tf.summary.image("num"+str(i), cm_image,step=num)