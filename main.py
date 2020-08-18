import tensorflow as tf
import tensorflow.keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import shutil
import random
from PIL import Image

train = False

paths = ["D:/images/viral", "D:/images/normal", "D:/images/bad"]

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

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
        im = cv2.resize(im, (120, 90))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im / 255.0
        return im


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (120,90))
    new_array = new_array.reshape(-1, 120, 90, 3)
    return new_array / 255.0

'''
for index, dir in enumerate(paths):
    traindir = dir + "/train/"
    testdir = dir + "/test/"

    x = 0

    for file in os.listdir(traindir):
        if type(convert_image(os.path.join(traindir, file))) == int:
            continue
        img = convert_image(os.path.join(traindir, file))
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, save_prefix="test", save_format="jpeg"):
            for z in batch:
                im = Image.fromarray((z * 255).astype(np.uint8)).convert("RGB")
                im.save(os.path.join(traindir, "test" + str(i + x) + file))
            i += 1
            if i > 20:
                break
        x += 30

    x = 0
    print(traindir + " done")

    for file in os.listdir(testdir):
        if type(convert_image(os.path.join(testdir, file))) == int:
            continue
        img = convert_image(os.path.join(testdir, file))
        img = img.reshape((1,) + img.shape)
        i = 0
        for batch in datagen.flow(img, save_prefix="test", save_format="jpeg"):
            for z in batch:
                im = Image.fromarray((z * 255).astype(np.uint8)).convert("RGB")
                im.save(os.path.join(testdir, "test" + str(i + x) + file))
            i += 1
            if i > 20:
                break
        x += 30

    print(testdir + " done")


print("Done")

'''

if train:
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

        print(traindir + " done")

        for file in os.listdir(testdir):
            if type(convert_image(os.path.join(testdir, file))) == int:
                continue
            test_image.append(convert_image(os.path.join(testdir, file)))
            test_label.append(index)

        print(testdir + " done")


    train_image = np.array(train_image)
    test_image = np.array(test_image)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    print(len(train_image))
    print(len(train_label))
    print(len(test_image))
    print(len(test_label))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(90, 120, 3)))
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
            monitor="val_accuracy",
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            mode='max')

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

if train:
    model.fit(train_image, train_label, epochs=1000, validation_data=(test_image, test_label), callbacks=[checkpoint_callback])

prediction = model.predict(prepare("test4.jpg"))
print(prediction)
