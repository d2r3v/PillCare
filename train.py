# train.py (Starter template)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)

train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    "pill_dataset",
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    "pill_dataset",
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

base_model = ResNet50(include_top=False, input_shape=(224,224,3), weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

model.save("pill_classifier.h5")
# This code is a simple image classification model using TensorFlow and Keras. It uses the ResNet50 architecture as a base model, adds a few layers on top, and trains it on a dataset of pill images. The dataset is split into training and validation sets using the ImageDataGenerator class. The model is then trained for 5 epochs and saved to a file named "pill_classifier.h5".
# The code uses the ResNet50 model as a base, which is a popular deep learning architecture for image classification tasks. The model is trained on images of pills, and the final output layer has a number of neurons equal to the number of classes in the dataset, with a softmax activation function to produce class probabilities.
# The model is compiled with the Adam optimizer and categorical crossentropy loss function, which is suitable for multi-class classification problems. The training process uses the fit method to train the model on the training data and validate it on the validation data.
# The model is saved to a file after training, which can be used later for inference or further training.
