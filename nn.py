import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

os.chdir('/content/drive/MyDrive/data.rar (Unzipped Files)/train')

image_df=pd.read_csv('new_data.csv')
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)


train_images = train_generator.flow_from_dataframe(
    dataframe=image_df,
    x_col='Filepath',
    y_col='Age',
    target_size=(150, 150),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)



callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

base = tf.keras.applications.Resnet152(weights='imagenet', input_shape=(150, 150, 3), include_top=False)

x = Flatten()(base.output)


outputs = tf.keras.layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=base.inputs, outputs=x)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    train_images,
    epochs=100,
)
model.save('resnet.h5')
