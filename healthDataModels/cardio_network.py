'''
Assignment 2 for CPSC 383
Fall 2024

Works on the heart.csv data set
Goal is to compare different variations on a basic neural net
in TensorBoard and write an analysis of the results

Author: Divyansh Rana
30117089
30 Oct, 2024
Tut T01
'''

# Suppress extra log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define TensorBoard log directory
log_dir = "logs/fit2/run_name/"  

data_stats = {
    "sbp": [138.3, 20.5],
    "tobacco": [3.64, 4.59],
    "ldl": [4.74, 2.07],
    "adiposity": [25.4, 7.77],
    "typea": [53.1, 9.81],
    "obesity": [26.0, 4.21],
    "alcohol": [17.0, 24.5]
}

# Load and preprocess data
def prepData():
    data = pd.read_csv("heart.csv")

    # Encode family history as binary and normalize age
    data['famhist'] = data['famhist'].apply(lambda x: 1 if x == "Present" else 0)
    data['age'] = data['age'] / 64  # Max age = 64

    # Standardize other columns
    for col in ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol']:
        mean, std = data_stats[col]
        data[col] = (data[col] - mean) / std

    # Split into features (column 1 to 10) and labels (last column)
    x = data.iloc[:, 1:-1].values
    y = data['chd'].values


    # Train-test split with 5:1 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/6)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = prepData()


# Test data dimensions
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# Model 1: Baseline
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir + "baseline_new", histogram_freq=1)
model.fit(x_train, y_train, epochs=20, callbacks=[tensorboard_callback])
model.evaluate(x_test, y_test, verbose=2)

# Model 2: Regularized with Dropout and Batch Normalization
model1 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dense(2)
])
model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tensorboard_callback1 = tf.keras.callbacks.TensorBoard(log_dir= log_dir + "ver1_new", histogram_freq=1)
early_stopping1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model1.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping1, tensorboard_callback1])
model1.evaluate(x_test, y_test, verbose=2)

# Model 3: Lighter Architecture with L2 Regularization
model2 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2)
])
model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tensorboard_callback2 = tf.keras.callbacks.TensorBoard(log_dir= log_dir + "ver2_new", histogram_freq=1)
model2.fit(x_train, y_train, epochs=25, validation_split=0.2, callbacks=[tensorboard_callback2])
model2.evaluate(x_test, y_test, verbose=2)

# Model 4: Enhanced Model with Deeper Layers and Dropout
model3 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
tensorboard_callback3 = tf.keras.callbacks.TensorBoard(log_dir= log_dir +"ver3_new", histogram_freq=1)
early_stopping3 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model3.fit(x_train, y_train, epochs=150, validation_split=0.2, callbacks=[early_stopping3, tensorboard_callback3])
model3.evaluate(x_test, y_test, verbose=2)

# Model 5: Simplified Network with Dropout
model4 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(9,)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2)
])
model4.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
tensorboard_callback4 = tf.keras.callbacks.TensorBoard(log_dir=log_dir +"ver4_new", histogram_freq=1)
early_stopping4 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model4.fit(x_train, y_train, epochs=60, validation_split=0.2, callbacks=[early_stopping4, tensorboard_callback4])
model4.evaluate(x_test, y_test, verbose=2)
