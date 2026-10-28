import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "C:\\Users\\Lenovo\\Downloads\\archive_cat_dog"
IMG_SIZE = 150
BATCH_SIZE = 32
datagen = ImageDataGenerator(
    rescale = 1.0/255,
    validation_split = 0.2
)

from google.colab import files
import zipfile

# Upload your dataset zip file
uploaded = files.upload()

# Example: if you uploaded archive_cat_dog.zip
with zipfile.ZipFile("archive_cat_dog.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

DATA_DIR = "/content/dataset/archive_cat_dog"

#Training
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    subset = "training"
)

#Validation
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    subset = "validation"
)

rsModel = ResNet50(
    weights = "imagenet",
    include_top = False,
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
)
rsModel.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(rsModel.output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
otp = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.models.Model(inputs=rsModel.input, outputs = otp)

model.compile(optimizer="adam",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

# Train the model
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=5)

#Plotting
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.legend()
plt.title("accuracy")
plt.show()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your local dataset
DATA_DIR = r"C:\Users\YourName\Desktop\PetImages"

IMG_SIZE = 150
BATCH_SIZE = 32

# Create ImageDataGenerator with normalization and validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training generator
train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',   # cats vs dogs → binary classification
    subset='training'
)

# Validation generator
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Example: Using ResNet50 pretrained on ImageNet
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze base model

# Add custom classification head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=5)