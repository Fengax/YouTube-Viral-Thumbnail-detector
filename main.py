import tensorflow as tf
import tensorflow.keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil
import random

paths = ["D:/images/viral", "D:/images/normal", "D:/images/bad"]

'''

for dir in paths:
    overall_path = []
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            overall_path.append(os.path.join(dir, file))
    test_sampling = random.sample(overall_path, 120)
    train_sampling = list(set(overall_path) - set(test_sampling))
    for i in test_sampling:
        filename = os.path.basename(i)
        shutil.move(i, os.path.join(dir + "/test/", filename))
    for i in train_sampling:
        filename = os.path.basename(i)
        shutil.move(i, os.path.join(dir + "/train/", filename))

'''


def convert_image(path):
    text = open(path, "rb")
    check_chars = text.read()[-2:]
    if check_chars != b'\xff\xd9':
        return 0
    else:
        im = cv2.imread(path)
        im = cv2.resize(im, (240, 180))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im / 255.0
        return im


train_image = []
test_image = []
train_label = []
test_label = []

for index, dir in enumerate(paths):
    traindir = dir + "/train/"
    testdir = dir + "/test/"

    for file in os.listdir(traindir):
        if type(convert_image(os.path.join(traindir, file))) == int:
            continue
        train_image.append(convert_image(os.path.join(traindir, file)))
        train_label.append(index)

    for file in os.listdir(testdir):
        if type(convert_image(os.path.join(testdir, file))) == int:
            continue
        test_image.append(convert_image(os.path.join(testdir, file)))
        test_label.append(index)


train_image = np.array(train_image)
test_image = np.array(test_image)
train_label = np.array(train_label)
test_label = np.array(test_label)

print(len(train_image))
print(len(train_label))
print(len(test_image))
print(len(test_label))



'''
(traind_image, traind_label), (testd_image, testd_label) = tf.keras.datasets.cifar10.load_data()

traind_image = traind_image / 255.0
testd_image = testd_image / 255.0

'''


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 240, 3)))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3))

model.compile(optimizer = 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor="loss",
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            mode='min')

model.fit(train_image, train_label, epochs=1000, validation_data=(test_image, test_label), callbacks=[checkpoint_callback])

