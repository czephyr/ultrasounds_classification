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

from tensorflow.keras.models import load_model

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

k_results = []
num = 0
arch_write_flag = True

cv = StratifiedGroupKFold(n_splits=5)
cv1 = StratifiedGroupKFold(n_splits=4)
learn_idxs, test_idxs = next(cv.split(dfNoOther["Filename"], dfNoOther["ml_class"], dfNoOther["patient"]))
learn = dfNoOther.iloc[learn_idxs]
test = dfNoOther.iloc[test_idxs]
train_idxs, val_idxs = next(cv1.split(learn["Filename"], learn["ml_class"], learn["patient"]))
train = learn.iloc[train_idxs]
val = learn.iloc[val_idxs]

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

classifier = load_model('currentModel/C4')

for layer in classifier.layers[:-3]:
    layer.trainable = False

classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

log_dir = 'tensorboard_logsPart2/noOtherTransferC'+str(NUM_TOTAL_CONVS)
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
        tf.summary.text("num"+str(NUM_TOTAL_CONVS),result_text,step=num)

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
        tf.summary.image("num"+str(NUM_TOTAL_CONVS), cm_image,step=num)