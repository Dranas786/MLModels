# CPSC 383, F24
# Python code to create a MNIST model and test it with my own handwritten code

# Author: Divyansh Rana, 30117089, Tut T01
# Code reference - https://www.youtube.com/watch?v=9cPMFTwBdM4

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras import layers, models
import numpy as np

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Organize the data into train and test with a ratio of 6:1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0  # Reshape and normalize training images
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0    # Reshape and normalize test images

# Convert labels to categorical format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the CNN model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # First Conv2D layer
model.add(layers.MaxPooling2D((2, 2)))                                             # First MaxPooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())                                                        # Flatten layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))                                  # Output layer

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the model
test_accuracy = model.evaluate(test_images, test_labels)[1]
print(f"Test Accuracy: {test_accuracy}")

# Load handwritten digits
handwritten_digits = np.load("digits.npy")
handwritten_digits[handwritten_digits == 255] = 0  # Adjust for default value differences
handwritten_digits = handwritten_digits.reshape(10, 28, 28, 1).astype("float32") / 255.0  # Reshape and normalize

# Predict for each handwritten digit
predictions = []
for digit in handwritten_digits:
    digit = np.expand_dims(digit, axis=0)  # Model expects a batch, so expand dimension
    prediction = model.predict(digit)
    predictions.append(prediction[0])

# Print predictions
for i, prediction in enumerate(predictions):
    print(f"Digit {i} Prediction: {prediction}")
    print(f"Model guessed: {np.argmax(prediction)} with probability {np.max(prediction)}")
