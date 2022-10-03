# %%
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
INPUT_SHAPE = TARGET_SIZE + (3,)
RESCALE_FACTOR = 1.0/255

# Architechture Varibales
NUM_CLASSES = 4
DENSE_UNITS = 128

# Processing Variables
BATCH_SIZE = 32
EPOCHS = 200

k_results = []
num = 0
arch_write_flag = True

cv = StratifiedGroupKFold(n_splits=5)
cv1 = StratifiedGroupKFold(n_splits=4)
learn_idxs, test_idxs = next(cv.split(jointsNoUnk["Filename"], jointsNoUnk["ml_class"], jointsNoUnk["patient"]))
learn = jointsNoUnk.iloc[learn_idxs]
test = jointsNoUnk.iloc[test_idxs]
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
    color_mode="rgb"
)

set_val = gen_val.flow_from_dataframe(
    val,
    directory="/home/msa-project/cropped",
    x_col='Filename',
    y_col='ml_class',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

set_test = gen_test.flow_from_dataframe(
    test,
    directory="/home/msa-project/cropped",
    x_col='Filename',
    y_col='ml_class',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb",
    shuffle = False
)

# %%
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

## Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
base_model.trainable = False ## Not trainable weights

from tensorflow.keras import layers, models

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(4096, activation='relu')
dense_layer_2 = layers.Dense(4096, activation='relu')
prediction_layer = layers.Dense(4, activation='softmax')


model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

# %%
log_dir = 'tensorboard_logs/vgg16FullArchRGB/fold' + str(1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

# Fit Kth Model
fitted_model = model.fit(
    set_train,
    epochs=EPOCHS,
    validation_data=set_val,
    callbacks=[tensorboard_callback]
)

results = model.evaluate(set_test)
result_text = "Loss: "+str(results[0])+" Accuracy: " + str(results[1])
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    with tf.name_scope("Test_Metrics"):
        tf.summary.text("Fold"+str(1), result_text, step=1)

set_test.reset()
y_pred = model.predict(set_test, set_test.n // BATCH_SIZE+1)
class_pred = np.argmax(y_pred, axis=1)
labels = set_test.class_indices
cm = confusion_matrix(set_test.classes, class_pred)
cm_disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=list(labels.keys()))
cm_disp.plot()
cm_image = plot_to_image(plt.gcf())

file_writer2 = tf.summary.create_file_writer(log_dir)
with file_writer2.as_default():
    with tf.name_scope("Test_Confusion_Matrix"):
        tf.summary.image("Fold"+str(1), cm_image, step=1)

classifier.save("currentModel/C4")